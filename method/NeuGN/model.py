

import math
from dataclasses import dataclass
from typing import Optional, Tuple
from NeuGN.encoders.graphGT_encoder import GraphsGPTEncoder, GraphsGPTEncoderOutput
from NeuGN.encoders.mpnn_encoder import GNN
from NeuGN.encoders.nag_encoder import NAGEncoder
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

        self.wk = nn.Linear(
            args.decoder_config.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )

        self.wv = nn.Linear(
            args.decoder_config.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )

        self.wo = nn.Linear(
            args.decoder_config.n_heads * self.head_dim,
            args.decoder_config.dim,
            bias=False
        )


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

        keys = xk
        values = xv

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
        # print(scores.shape)
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

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class NodeIdEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        ne = torch.empty(max_len, d_model)
        torch.nn.init.orthogonal_(ne)
        ne.require_grad = False
        
        self.register_buffer('ne', ne)  
        self.num_embeddings = max_len 
        self.embedding_dim = d_model   

    def forward(self, node_ids):
        return self.ne[node_ids]


class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.vocab_size = params.decoder_config.vocab_size
        self.n_layers = params.decoder_config.n_layers
        self.subnodeid_size = params.decoder_config.sub_node_id_size


        self.tok_embeddings = nn.Embedding(
            params.decoder_config.vocab_size, params.decoder_config.dim,
        )

        self.node_embeddings = NodeIdEmbedding(
             params.decoder_config.dim, params.decoder_config.sub_node_id_size,
        )

        self.pos_embeddings = PositionalEmbedding(
             params.decoder_config.dim, params.decoder_config.pos_size,
        )

        self.type_embeddings = nn.Embedding(2, params.decoder_config.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.decoder_config.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.decoder_config.dim, eps=params.decoder_config.norm_eps)
        # self.output = ColumnParallelLinear(
        #     params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        # )
        self.output = nn.Sequential(
            nn.Linear(params.decoder_config.dim, params.decoder_config.dim),
            nn.GELU(),
            nn.Linear(params.decoder_config.dim, params.decoder_config.vocab_size),
        )


    def forward(self, graph_features: torch.Tensor, tokens: torch.Tensor, subnode_ids: torch.Tensor, token_mask_len: torch.Tensor, start_pos: int):
        if tokens is not None:
            _bsz, seqlen = tokens.shape
            h_tokens = self.tok_embeddings(tokens)
            h_tokens_subnode = self.node_embeddings(subnode_ids)
            token_types = torch.zeros_like(tokens, dtype=torch.long, device=tokens.device)
            h_tokens = h_tokens + self.type_embeddings(token_types) + h_tokens_subnode
            graph_feature_types = torch.ones(graph_features.size(0), graph_features.size(1), dtype=torch.long, device=graph_features.device)
            h_graph = graph_features + self.type_embeddings(graph_feature_types)
            h = torch.cat((h_graph, h_tokens), dim=1)
            seqlen += graph_features.shape[1]
        else:
            h = graph_features + self.type_embeddings(torch.ones(graph_features.size(0), graph_features.size(1), dtype=torch.long, device=graph_features.device))
            seqlen = h.shape[1]

        h = h + self.pos_embeddings(h)

        positions = torch.arange(seqlen, device=graph_features.device).unsqueeze(0)
        positions_repeated = positions.repeat(_bsz, 1) 
        token_mask_len_uns = token_mask_len.unsqueeze(1)
        valid_mask = positions_repeated < (token_mask_len_uns+1) 
        mask = torch.full_like(valid_mask, float('-inf'), device=graph_features.device).type_as(h)
        
        mask[valid_mask] = 0.0
        mask = mask.unsqueeze(2).repeat(1 , 1, seqlen).unsqueeze(1)


        for layer in self.layers:
            h = layer(h, start_pos, mask)
        h = self.norm(h)
        h_selected = h[:, 1:2, :]
        output = self.output(h_selected).float()
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
        
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.norm = RMSNorm(self.hidden_dim, eps=params.decoder_config.norm_eps)

    def forward(self, x):
        out = F.relu(self.input_layer(x.squeeze(1)))
        for layer in self.hidden_layers:
            residual = out.clone()
            out = F.silu(layer(out))
            out = residual + self.norm(out)
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
        elif self.encoder_type == 'nagphormer':
            self.encoder = NAGEncoder(params.encoder_config)
            
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
        elif self.encoder_type == 'nagphormer':
            graph_features = self.encoder(batch_graphs)
            graph_features = graph_features.unsqueeze(dim=1)
        return graph_features
    
    def get_decoder_output(self, graph_features, tokens: torch.Tensor, subnode_ids: torch.Tensor, token_mask_len: torch.Tensor,):
        if self.decoder_type == 'llama':
            output = self.decoder(graph_features, tokens, subnode_ids, token_mask_len, 0)
        elif self.decoder_type == 'mlp':
            output = self.decoder(graph_features)
        return output


    def forward(self, batch_graphs, tokens: torch.Tensor, subnode_ids: torch.Tensor, token_mask_len: torch.Tensor, start_pos: int, device):
        if self.encoder_type == 'gt':
            encoder_output:GraphsGPTEncoderOutput = self.encoder(batch_graphs, device)
            graph_features = encoder_output.fingerprint_tokens
            # graph_features = encoder_output.inputs_embeds
        elif self.encoder_type in self.gnn_list:
            graph_features = self.encoder(batch_graphs)
            graph_features = graph_features.unsqueeze(dim=1)
        elif self.encoder_type == 'nagphormer':
            graph_features = self.encoder(batch_graphs)
            graph_features = graph_features.unsqueeze(dim=1)

        if self.decoder_type == 'llama':
            output = self.decoder(graph_features, tokens, subnode_ids, token_mask_len, start_pos)
        elif self.decoder_type == 'mlp':
            output = self.decoder(graph_features)
        return output



