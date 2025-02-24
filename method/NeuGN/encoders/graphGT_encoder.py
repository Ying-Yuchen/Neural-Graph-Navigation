import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import warnings
from dataclasses import dataclass
from torch import nn, no_grad
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging, ModelOutput
from typing import List, Optional, Tuple, Union

import dgl
import ipdb

@dataclass
class GraphPositionEmbeddingOutput(ModelOutput):
    graph_position_embeds: torch.FloatTensor = None
    graph_position_features: Optional[torch.FloatTensor] = None
    orthonormal_features: Optional[torch.FloatTensor] = None
    embedding_ids: Optional[torch.FloatTensor] = None


@dataclass
class GraphsGPTEncoderOutput(ModelOutput):
    fingerprint_tokens: torch.FloatTensor = None

    inputs_embeds: Optional[torch.FloatTensor] = None
    identifier_embeds: Optional[torch.FloatTensor] = None
    graph_position_embeds: Optional[torch.FloatTensor] = None
    graph_position_features: Optional[torch.FloatTensor] = None
    orthonormal_features: Optional[torch.FloatTensor] = None
    graph_embedding_ids: Optional[torch.LongTensor] = None

    attention_mask: Optional[torch.Tensor] = None

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
@no_grad()
def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype = torch.float32,
        device: Union[torch.device, str] = "cuda",
):
    """
    Make causal mask used for bidirectional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)  # shape(tgt_len,)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


# Modified from transformers.models.bart.modeling_bart._make_causal_mask
@no_grad()
def _make_graph_causal_mask(
        input_shape: torch.Size,
        identifier_ids: torch.BoolTensor,
        num_fingerprint_tokens: int,
        share_fingerprint_tokens: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Union[torch.device, str] = "cuda",
):
    """
    Make causal mask used for bidirectional self-attention with graph token.
    """
    # common lower triangular matrix mask
    bsz, tgt_len = input_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(tgt_len, device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(tgt_len, 1), 0)
    mask = mask.unsqueeze(0).repeat(bsz, 1, 1)

    # unmask before node tokens
    # A. the first node token should see the latter two tokens
    # B. the edge token before each node token should see the node token
    mole_seq_len = tgt_len - num_fingerprint_tokens - 1

    if mole_seq_len > 0:
        full_indices_mask = torch.arange(mole_seq_len, device=device).unsqueeze(0).expand(bsz, mole_seq_len)

        # (A) unmask for the first node token, it should see the latter two tokens
        # get unmask position indices for node tokens
        indices_batch = torch.arange(bsz, device=device).repeat_interleave(2, dim=0)
        indices_row = torch.full((2 * bsz,), num_fingerprint_tokens + 1, device=device)
        indices_column = torch.arange(num_fingerprint_tokens + 2, num_fingerprint_tokens + 4, step=1, device=device).unsqueeze(0).expand(bsz, 2).flatten()

        mask[indices_batch, indices_row, indices_column] = 0

        # (B) unmask before node tokens
        # make no change on the BOS token, i.e., the BOS token won't see the first node token
        identifier_ids = identifier_ids.clone()
        identifier_ids[:, 0] = False  # skip the first node token

        # get unmask position indices for node tokens
        node_num = identifier_ids.sum(dim=1)
        indices_batch = torch.arange(bsz, device=device).repeat_interleave(node_num, dim=0)
        indices_row = full_indices_mask[identifier_ids] + num_fingerprint_tokens
        indices_column = indices_row + 1

        mask[indices_batch, indices_row, indices_column] = 0

    # unmask within fingerprint tokens
    if share_fingerprint_tokens:
        mask[:, :, :num_fingerprint_tokens] = 0  # fingerprint tokens can see each other

    mask = mask.to(dtype)
    # print(mask[0, :, :].clone().cpu().numpy())

    return mask[:, None, :, :]  # (bsz, 1, tgt_len, tgt_len)


# Copied from transformers.models.bart.modeling_bart._expand_mask
@no_grad()
def _expand_mask(
        mask: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        tgt_len: int = None
):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    # inverted_mask = expanded_mask

    final_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    # ipdb.set_trace()
    return final_mask

    # return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class StableEmbedding(nn.Embedding):
    """
    Stable embedding from https://github.com/TimDettmers/bitsandbytes/blob/18e827d666fa2b70a12d539ccedc17aa51b2c97c/bitsandbytes/nn/modules.py#L21
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs) -> None:
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.norm = torch.nn.LayerNorm(embedding_dim)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight)
        self._fill_padding_idx_with_zero()

    def forward(self, input: torch.Tensor, offsets: Optional[torch.Tensor] = None, per_sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        emb = emb.to(torch.get_default_dtype())  # always apply layer norm in full precision
        return self.norm(emb).to(self.weight.dtype)

def generate_orthogonal_matrix(dim):
    # Step 1: Generate a random matrix
    random_matrix = torch.randn(dim, dim)
    # Step 2: Perform QR decomposition
    q, r = torch.linalg.qr(random_matrix)
    # Step 3: Return the orthogonal matrix Q
    return q


class GraphPositionStableEmbedding(nn.Module):

    def __init__(self, feature_dim, embedding_dim) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        orthogonal_tensor = generate_orthogonal_matrix(feature_dim)
        self.orthonormal_features = nn.Embedding(self.feature_dim, self.feature_dim)
        with torch.no_grad():
            self.orthonormal_features.weight.copy_(orthogonal_tensor)
        self.orthonormal_features.weight.requires_grad = False
        # self.orthonormal_features = nn.Parameter(orthogonal_tensor, requires_grad=False)
        # self.learnable_orthonormal_features = nn.Embedding(feature_dim, feature_dim)
        self.graph_position_proj = nn.Linear(2 * feature_dim, embedding_dim, bias=False)
        self.norm = torch.nn.LayerNorm(embedding_dim)

    def forward(
            self,
            graph_position_ids_1,
            graph_position_ids_2,
            identifier_ids,
            dtype,
            device,
            use_random_id=False,
            adaptive_position_length=False,  # useful only when "use_random_id" is "True"
            embedding_ids=None,
            orthonormal_features=None,
            graph_position_features=None,
            return_features=True,
    ) -> GraphPositionEmbeddingOutput:
        """
        Graph embedding modified from https://github.com/jw9730/tokengt
        Stable embedding from https://github.com/TimDettmers/bitsandbytes/blob/18e827d666fa2b70a12d539ccedc17aa51b2c97c/bitsandbytes/nn/modules.py#L21
        """
        if graph_position_features is None:
            batch_size, graph_seq_len = identifier_ids.shape  # also the shape of "graph_position_ids_1" and "graph_position_ids_2"
            max_node_cnt = int(torch.max(torch.sum(identifier_ids.clone(), dim=1)).item())

            # (batch_size, max_node_cnt, position_feature_size)
            if orthonormal_features is None:
                if embedding_ids is None:
                    if use_random_id:  # randomly assign positional embeddings to atoms
                        if adaptive_position_length:  # indices range between (0, max_node_cnt)
                            _, embedding_ids = torch.rand((batch_size, max_node_cnt), device=device).sort(dim=1)  # random indices
                        else:  # indices range between (0, feature_dim)
                            _, embedding_ids = torch.rand((batch_size, self.feature_dim), device=device).sort(dim=1)  # random indices
                            embedding_ids = embedding_ids[:, :max_node_cnt]
                    else:
                        embedding_ids = torch.arange(0, max_node_cnt, device=device).expand(batch_size, max_node_cnt)  # incremental indices
                orthonormal_features = self.orthonormal_features(embedding_ids)
            
            start_indices = torch.arange(0, batch_size * max_node_cnt, step=max_node_cnt, device=device).unsqueeze(1)  # (batch_size, 1)
            graph_position_features_1_indices = (graph_position_ids_1 + start_indices).reshape(-1)  # (batch_size * graph_seq_len)
            graph_position_features_2_indices = (graph_position_ids_2 + start_indices).reshape(-1)  # (batch_size * graph_seq_len)

            reshaped_orthonormal_features = orthonormal_features.reshape(-1, self.feature_dim)  # (batch_size * max_node_cnt, feature_dim)
            graph_position_features_1 = reshaped_orthonormal_features[graph_position_features_1_indices].reshape(batch_size, graph_seq_len, self.feature_dim)  # (batch_size, graph_seq_len, feature_dim)
            graph_position_features_2 = reshaped_orthonormal_features[graph_position_features_2_indices].reshape(batch_size, graph_seq_len, self.feature_dim)  # (batch_size, graph_seq_len, feature_dim)
            graph_position_features = torch.cat((graph_position_features_1, graph_position_features_2), dim=2)  # (batch_size, graph_seq_len, 2 * feature_dim)
            # ipdb.set_trace()
        graph_position_embeds = self.graph_position_proj(graph_position_features)  # (batch_size, graph_seq_len, embedding_dim)
        graph_position_embeds = self.norm(graph_position_embeds).to(dtype)

        return GraphPositionEmbeddingOutput(
            graph_position_embeds=graph_position_embeds,
            graph_position_features=graph_position_features if return_features else None,
            orthonormal_features=orthonormal_features if return_features else None,
            embedding_ids=embedding_ids if return_features else None,
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class GraphsGPTMLP(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        super().__init__()
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class GraphsGPTSelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.encoder_hidden_size
        self.num_heads = config.num_encoder_heads
        self.head_dim = self.hidden_size // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_query_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.size()

        # TODO: the cache logic is bugged, need a fix
        if past_query_key_value is None:
            query_states = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            past_seq_len = past_query_key_value[0].shape[-2]
            this_seq_len = seq_len - past_seq_len

            this_hidden_states = hidden_states[:, past_seq_len:, :]
            query_states = self.q_proj(this_hidden_states).view(bsz, this_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(this_hidden_states).view(bsz, this_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(this_hidden_states).view(bsz, this_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            # [bsz, nh, sl, hd]

            # reuse k, v, self_attention
            query_states = torch.cat([past_query_key_value[0], query_states], dim=2)
            key_states = torch.cat([past_query_key_value[1], key_states], dim=2)
            value_states = torch.cat([past_query_key_value[2], value_states], dim=2)

        past_query_key_value = (query_states, key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, seq_len, seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, seq_len, seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, seq_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_query_key_value


class GraphsGPTEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.encoder_hidden_size
        self.self_attn = GraphsGPTSelfAttention(config=config)
        self.mlp = GraphsGPTMLP(
            hidden_size=self.hidden_size,
            # intermediate_size=config.intermediate_size,
            intermediate_size=4 * self.hidden_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)  # Pre-Normalization for Self Attention

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_query_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)  # Pre-Normalization for Fully Connected
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GraphsGPTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.padding_idx = config.pad_token_id
        # self.atom_vocab_size = config.atom_vocab_size
        # self.bond_vocab_size = config.bond_vocab_size
        self.position_feature_size = config.position_feature_size
        self.node_feature_size = config.node_feature_size
        self.edge_feature_size = config.edge_feature_size

        # basic embeddings
        # 0 for padding token, 1 for B0S token
        # Although the encoder doesn't receive BOS token as the input, we still leave its position in case sharing embedding with the decoder.
        # self.embed_tokens = StableEmbedding(2 + config.atom_vocab_size + config.bond_vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        # 0 for bonds (edges), 1 for atoms (nodes)
        self.embed_identifier = StableEmbedding(2, config.encoder_hidden_size)
        self.linear_node = nn.Linear(self.node_feature_size, config.encoder_hidden_size)
        self.linear_edge = nn.Linear(self.edge_feature_size, config.encoder_hidden_size)

        # fingerprint embeddings
        assert config.num_fingerprints > 0
        self.embed_fingerprint = StableEmbedding(1, config.encoder_hidden_size)
        self.embed_fingerprint_position = StableEmbedding(config.num_fingerprints, config.encoder_hidden_size)

        # graph position embeddings
        self.embed_graph_position = GraphPositionStableEmbedding(config.position_feature_size, config.encoder_hidden_size)

        # create layers
        self.encoder_layers = nn.ModuleList([GraphsGPTEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.encoder_norm = RMSNorm(config.encoder_hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = config.gradient_checkpointing

    def _prepare_encoder_attention_mask(
            self,
            attention_mask,
            input_shape,
            dtype=torch.float32,
            device="cuda",
    ):
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, dtype=dtype, tgt_len=input_shape[-1]).to(device)
        return expanded_attn_mask
    
    '''
    graph gt encoder pre-process
    '''
    def process_batched_graph(self, batched_graph):

        batch_node_count = batched_graph.batch_num_nodes()
        node_num_all = len(batched_graph.nodes())
        max_node_num = int(torch.max(batch_node_count))
        batch_edge_count = batched_graph.batch_num_edges()
        max_edge_num = int(torch.max(batch_edge_count))
        max_edge_num /= 2 #若是无向边
        
        graph_list = dgl.unbatch(batched_graph)

        batch_size = len(batch_node_count)
        total_length = int(max_node_num + max_edge_num)

        batched_feature_ids = torch.zeros((batch_size, total_length), dtype=torch.long)
        batched_mask = torch.zeros((batch_size, total_length), dtype=torch.long)
        batched_identity_mask = torch.zeros((batch_size, total_length), dtype=torch.long)
        batched_position_ids1 = torch.zeros((batch_size, total_length), dtype=torch.long)
        batched_position_ids2 = torch.zeros((batch_size, total_length), dtype=torch.long)
        start_node_id, start_edge_id = 0, 0

        for graph_id, (node_num, edge_num, graph) in enumerate(zip(batch_node_count, batch_edge_count, graph_list)):
            position_ids1, position_ids2, identity, mask = [], [], [], []  ### identity: 1 refers to node
            node_ids = graph.nodes()
            position_ids1 += node_ids
            position_ids2 += node_ids
            src_nodes, dst_nodes, edge_ids = graph.edges(form='all')
            src_nodes_clean, dst_nodes_clean, edge_ids_clean = [], [], []
            node_ids_batched = [node_id + start_node_id for node_id in node_ids]
            for src_node, dst_node, edge_id in zip(src_nodes, dst_nodes, edge_ids):
                if src_node <= dst_node:
                    src_nodes_clean.append(src_node)
                    dst_nodes_clean.append(dst_node)
                    edge_ids_clean.append(edge_id)
            edge_ids_batched = [edge_id + start_edge_id + node_num_all for edge_id in edge_ids_clean]
            feature_ids = node_ids_batched + edge_ids_batched
            # feature_ids = node_ids_batched
            position_ids1 += src_nodes_clean
            position_ids2 += dst_nodes_clean
            identity += [1] * len(node_ids)
            mask += [1] * len(position_ids1)

            start_node_id += node_num
            start_edge_id += edge_num

            batched_feature_ids[graph_id][0:len(feature_ids)] = torch.tensor(feature_ids)
            batched_mask[graph_id][0:len(mask)] = torch.tensor(mask)
            batched_identity_mask[graph_id][0:len(identity)] = torch.tensor(identity)
            # batched_identity_mask[graph_id][0:len(node_ids)] = torch.tensor(identity)  ###若仅仅使用node feat
            batched_position_ids1[graph_id][0:len(position_ids1)] = torch.tensor(position_ids1)
            batched_position_ids2[graph_id][0:len(position_ids2)] = torch.tensor(position_ids2)

        return batched_feature_ids, batched_mask, batched_identity_mask, batched_position_ids1, batched_position_ids2

    def forward(self, batched_graph, device):
        batched_feature_ids, batched_mask, batched_identity_mask, batched_position_ids1, batched_position_ids2 = self.process_batched_graph(batch_graphs)
        batched_feature_ids, batched_mask, batched_identity_mask, batched_position_ids1, batched_position_ids2 = batched_feature_ids.to(device), batched_mask.to(device), batched_identity_mask.to(device), batched_position_ids1.to(device), batched_position_ids2.to(device)
        encoder_output = self.forward_encoder(batch_graphs, batched_feature_ids, batched_position_ids1, batched_position_ids2, batched_identity_mask, self.finger_num, batched_mask)
        return encoder_output

    def forward_encoder(
            self,
            batched_graphs: dgl.DGLGraph = None,
            input_ids: torch.LongTensor = None,
            graph_position_ids_1: torch.LongTensor = None,
            graph_position_ids_2: torch.LongTensor = None,
            identifier_ids: torch.BoolTensor = None,

            num_fingerprint_tokens: Optional[int] = None,
            attention_mask: Optional[torch.Tensor] = None,  # padding mask

            inputs_embeds: Optional[torch.FloatTensor] = None,
            identifier_embeds: Optional[torch.FloatTensor] = None,
            graph_position_embeds: Optional[torch.FloatTensor] = None,
            graph_position_features: Optional[torch.FloatTensor] = None,
            orthonormal_features: Optional[torch.FloatTensor] = None,
            graph_embedding_ids: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> GraphsGPTEncoderOutput:

        batch_size, seq_length = identifier_ids.shape
        device = identifier_ids.device

        node_features = self.linear_node(batched_graphs.ndata['feat'])
        edge_features = self.linear_edge(batched_graphs.edata['feat'])
        node_edge_features = torch.cat((node_features, edge_features), dim=0)
        # ipdb.set_trace()
        # input check
        if input_ids is not None and inputs_embeds is not None:
            assert input_ids.shape[:2] == inputs_embeds.shape[:2]  # ids and embeds must have the same length
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # input check
        if graph_position_ids_1 is not None and graph_position_ids_2 is not None:
            assert graph_position_ids_1.shape[:2] == graph_position_ids_2.shape[:2]  # both ids must have the same length
            if graph_position_embeds is not None:
                assert graph_position_ids_1.shape[:2] == graph_position_embeds.shape[:2]  # ids and embeds must have the same length
        elif graph_position_ids_1 is None and graph_position_ids_2 is None and graph_position_embeds is None:
            raise ValueError("You have to specify either graph_position_ids or graph_position_embeds")
        else:
            raise ValueError("graph_position_ids have to be either both specified or neither specified.")

        # set the number of encoded fingerprints
        if num_fingerprint_tokens is None:
            num_fingerprint_tokens = self.config.num_fingerprints

        # get encoder embeds
        fingerprint_embeds = self.embed_fingerprint(
            torch.zeros(batch_size, num_fingerprint_tokens, dtype=torch.int64, device=device)
        )  # (batch_size, num_fingerprint_tokens, embed_dim)
        fingerprint_position_embeds = self.embed_fingerprint_position(
            torch.arange(0, num_fingerprint_tokens, device=device).expand(batch_size, num_fingerprint_tokens)
        )  # (batch_size, num_fingerprint_tokens, embed_dim)

        if inputs_embeds is None:
            # inputs_embeds = self.embed_tokens(input_ids)  # (batch_size, seq_len, embed_dim)
            inputs_embeds = node_edge_features[input_ids]
        dtype = inputs_embeds.dtype

        if graph_position_embeds is None:
            graph_embedding_outputs: GraphPositionEmbeddingOutput = self.embed_graph_position(
                graph_position_ids_1,
                graph_position_ids_2,
                identifier_ids,
                dtype=dtype,
                device=device,
                use_random_id=True,
                adaptive_position_length=self.config.adaptive_position_length,
                embedding_ids=graph_embedding_ids,
                orthonormal_features=orthonormal_features,
                graph_position_features=graph_position_features,
                return_features=False,
            )
            graph_position_embeds = graph_embedding_outputs.graph_position_embeds  # (batch_size, seq_len, embed_dim)
            graph_position_features = graph_embedding_outputs.graph_position_features  # None
            orthonormal_features = graph_embedding_outputs.orthonormal_features  # None
            graph_embedding_ids = graph_embedding_outputs.embedding_ids  # None

        if identifier_embeds is None:
            identifier_embeds = self.embed_identifier(identifier_ids.clone().int())  # (batch_size, seq_len, embed_dim)

        # add embeds together and get hidden_states
        fingerprint_tokens = fingerprint_embeds + fingerprint_position_embeds  # (batch_size, num_fingerprint_tokens, embed_dim)
        molecule_tokens = inputs_embeds + graph_position_embeds + identifier_embeds  # (batch_size, seq_len, embed_dim)
        # molecule_tokens = inputs_embeds + identifier_embeds

        hidden_states = torch.cat((fingerprint_tokens, molecule_tokens), dim=1)  # (batch_size, num_fingerprint_tokens + seq_len, embed_dim)

        # get attention masks
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, num_fingerprint_tokens + seq_length), dtype=torch.bool, device=device)
        else:  # the attention mask is shaped like (batch_size, mole_seq_len) so we need to extend its dimension
            extra_dim = num_fingerprint_tokens + seq_length - attention_mask.shape[1]
            if extra_dim > 0:  # adding extra dimensions to the attention mask
                # extra_attention_mask = torch.ones((batch_size, extra_dim), dtype=torch.bool, device=device)
                extra_attention_mask = torch.ones((batch_size, extra_dim), dtype=torch.bool, device=device)
                attention_mask = torch.cat((extra_attention_mask, attention_mask), dim=1)
            else:
                attention_mask = attention_mask

        attention_mask = self._prepare_encoder_attention_mask(
            attention_mask,
            (batch_size, num_fingerprint_tokens + seq_length),
            dtype=dtype,
            device=device,
        )

        # forward encoder
        for idx, encoder_layer in enumerate(self.encoder_layers):
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                )
            else:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                )

        hidden_states = self.encoder_norm(hidden_states)  # (batch_size, num_fingerprint_tokens + seq_len, embed_dim)

        # get encoded fingerprint tokens
        fingerprint_tokens = hidden_states[:, :num_fingerprint_tokens, :]  # (batch_size, num_fingerprint_tokens, embed_dim)
        # fingerprint_tokens = hidden_states[:, :, :]

        return GraphsGPTEncoderOutput(
            fingerprint_tokens=fingerprint_tokens,
            inputs_embeds=inputs_embeds,
            identifier_embeds=identifier_embeds,
            graph_position_embeds=graph_position_embeds,
            graph_position_features=graph_position_features,
            orthonormal_features=orthonormal_features,
            graph_embedding_ids=graph_embedding_ids,
            attention_mask=attention_mask,
        )