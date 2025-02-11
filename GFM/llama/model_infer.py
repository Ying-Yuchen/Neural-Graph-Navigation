# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple
from llama.encoders.graphGT_encoder import GraphsGPTEncoder, GraphsGPTEncoderOutput
from llama.encoders.mpnn_encoder import GNN
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn
from dgl.nn import GraphConv, GATConv, GINConv
import dgl
import ipdb
import box


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_kv_heads = args.decoder_config.n_heads if args.decoder_config.n_kv_heads is None else args.decoder_config.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        model_parallel_size = 1
        self.n_local_heads = args.decoder_config.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.decoder_config.dim // args.decoder_config.n_heads
        self.wq = nn.Linear(
            args.decoder_config.dim,
            args.decoder_config.n_heads * self.head_dim,
            bias=False
        )
        # self.wq = ColumnParallelLinear(
        #     args.dim,
        #     args.n_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        self.wk = nn.Linear(
            args.decoder_config.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        # self.wk = ColumnParallelLinear(
        #     args.dim,
        #     self.n_kv_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        self.wv = nn.Linear(
            args.decoder_config.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        # self.wv = ColumnParallelLinear(
        #     args.dim,
        #     self.n_kv_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        self.wo = nn.Linear(
            args.decoder_config.n_heads * self.head_dim,
            args.decoder_config.dim,
            bias=False
        )
        # self.wo = RowParallelLinear(
        #     args.n_heads * self.head_dim,
        #     args.dim,
        #     bias=False,
        #     input_is_parallel=True,
        #     init_method=lambda x: x,
        # )
        self.cache_k = torch.zeros(
            (
                args.decoder_config.max_batch_size,
                args.decoder_config.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.decoder_config.max_batch_size,
                args.decoder_config.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        # freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        ###原始代码，使用缓存获得keys 和 values
        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)

        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # keys = self.cache_k[:bsz, : start_pos + seqlen]
        # values = self.cache_v[:bsz, : start_pos + seqlen]
        ###

        ###新代码，直接获得
        keys = xk
        values = xv
        ###

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # self.w1 = ColumnParallelLinear(
        #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        # )
        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        # self.w2 = RowParallelLinear(
        #     hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        # )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        # self.w3 = ColumnParallelLinear(
        #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        # )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args):
        super().__init__()
        self.n_heads = args.decoder_config.n_heads
        self.dim = args.decoder_config.dim
        self.head_dim = args.decoder_config.dim // args.decoder_config.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.decoder_config.dim,
            hidden_dim=4 * args.decoder_config.dim,
            multiple_of=args.decoder_config.multiple_of,
            ffn_dim_multiplier=args.decoder_config.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.decoder_config.dim, eps=args.decoder_config.norm_eps)
        self.ffn_norm = RMSNorm(args.decoder_config.dim, eps=args.decoder_config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos,  mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.vocab_size = params.decoder_config.vocab_size
        self.n_layers = params.decoder_config.n_layers

        # self.tok_embeddings = VocabParallelEmbedding(
        #     params.vocab_size, params.dim, init_method=lambda x: x
        # )
        self.tok_embeddings = nn.Embedding(
            params.decoder_config.vocab_size, params.decoder_config.dim,
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.decoder_config.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.decoder_config.dim, eps=params.decoder_config.norm_eps)
        # self.output = ColumnParallelLinear(
        #     params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        # )
        self.output = nn.Linear(
            params.decoder_config.dim, params.decoder_config.vocab_size, bias=False
        )


        # self.freqs_cis = precompute_freqs_cis(
        #     params.decoder_config.dim // params.decoder_config.n_heads,
        #     params.decoder_config.max_seq_len * 2,
        #     params.decoder_config.rope_theta,
        # ).to(params.device)


    def forward(self, graph_features: torch.Tensor, tokens: torch.Tensor, start_pos: int):
        # print('000000')
        if tokens is not None:
            _bsz, seqlen = tokens.shape
            
            h = self.tok_embeddings(tokens)

            h = torch.cat((graph_features, h), dim=1)
            seqlen += graph_features.shape[1]
        else:
            h = graph_features
            seqlen = h.shape[1]

        # self.freqs_cis = self.freqs_cis.to(h.device)
        # freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
    
    
class ResidualMLP(nn.Module):
    def __init__(self, params):
        super(ResidualMLP, self).__init__()
        
        self.input_dim = params.decoder_config.dim
        self.hidden_dim = params.decoder_config.dim
        self.output_dim = params.decoder_config.vocab_size
        self.n_layers = params.decoder_config.n_layers
        
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Define hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layers-1)]
        )
        
        # Define the output layer
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.norm = RMSNorm(self.hidden_dim, eps=params.decoder_config.norm_eps)

    def forward(self, x):
        # Input layer
        out = F.relu(self.input_layer(x.squeeze(1)))
        
        # Pass through hidden layers with residual connections
        for layer in self.hidden_layers:
            # Save input for residual connection
            residual = out.clone()
            # Apply layer and activation
            out = F.silu(layer(out))
            # Add the residual
            out = residual + self.norm(out)
        # Output layer
        out = self.output_layer(out)
        return out
    
    
class GraphDecoder(nn.Module):
    def __init__(self, params):
        super(GraphDecoder, self).__init__()
        # self.gnn = GNN(params)
        self.encoder_type = params.encoder_config.encoder_name ### gt or gnn
        self.decoder_type = params.decoder_config.decoder_type
        
        self.gnn_list = ['gat', 'gcn', 'gin']
        if self.encoder_type in self.gnn_list:
            self.encoder = GNN(params)
        elif self.encoder_type == 'gt':
            self.encoder = GraphsGPTEncoder(params)
            
        if self.decoder_type == 'llama':
            self.decoder = Transformer(params)
        elif self.decoder_type == 'mlp':
            self.decoder =  ResidualMLP(params)
        self.dim = params.decoder_config.dim
        self.finger_num = params.gt_config.num_fingerprints
        
    def get_encoder_tensor(self, batch_graphs, device):
        if self.encoder_type == 'gt':
            encoder_output = self.encoder(batch_graphs, device)
            graph_features = encoder_output.fingerprint_tokens
            # graph_features = encoder_output.inputs_embeds
        elif self.encoder_type in self.gnn_list:
            graph_features = self.encoder(batch_graphs)
            graph_features = graph_features.unsqueeze(dim=1)
        return graph_features
    
    def get_decoder_output(self, graph_features, tokens: torch.Tensor):
        if self.decoder_type == 'llama':
            output = self.decoder(graph_features, tokens, 0)
        elif self.decoder_type == 'mlp':
            output = self.decoder(graph_features)
        return output


    def forward(self, batch_graphs, tokens: torch.Tensor, start_pos: int, device):
        if self.encoder_type == 'gt':
            encoder_output:GraphsGPTEncoderOutput = self.encoder(batch_graphs, device)
            graph_features = encoder_output.fingerprint_tokens
            # graph_features = encoder_output.inputs_embeds
        elif self.encoder_type in self.gnn_list:
            graph_features = self.encoder(batch_graphs)
            graph_features = graph_features.unsqueeze(dim=1)
        # batch_size = tokens.shape[0]
        # graph_features = torch.zeros(batch_size, self.dim).cuda()
        if self.decoder_type == 'llama':
            output = self.decoder(graph_features, tokens, start_pos)
        elif self.decoder_type == 'mlp':
            output = self.decoder(graph_features)
        return output



