from torch import nn
from dgl.nn import GraphConv, GATConv, GINConv
import dgl
import torch
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, params):
        super(GNN, self).__init__()
        self.in_dim = params.encoder_config.graph_feature_dim
        self.out_dim = params.decoder_config.dim
        self.num_layers = params.encoder_config.encoder_layers
        encoder_name = params.encoder_config.encoder_name
        self.convs = nn.ModuleList()
        
        encoder_name = params.encoder_config.encoder_name.lower()
        
        if encoder_name == 'gcn':
            self.layer_type = GraphConv
        elif encoder_name == 'gat':
            self.layer_type = GATConv
        elif encoder_name == 'gin':
            self.layer_type = lambda in_dim, out_dim: GINConv(
                nn.Linear(in_dim, out_dim), 'sum'
            )  # Assuming 'mean' aggregator for GIN
        else:
            raise ValueError(f"Unsupported encoder name: {params.encoder_config.encoder_name}")

        # Add the first layer
        self.convs.append(self.layer_type(self.in_dim, self.out_dim))
        
        # Initialize the value embedding
        self.value_embedding = nn.Embedding(
            params.encoder_config.graph_value_num, self.in_dim
        )
        
        # Add additional layers
        for _ in range(1, self.num_layers):
            self.convs.append(self.layer_type(self.out_dim, self.out_dim))
        


    def forward(self, batched_graph):
        # Assuming the node features are stored in 'feat'
        h_id = batched_graph.ndata['feat_id']
        h = self.value_embedding(h_id)
        # Apply each GraphConv layer
        for conv in self.convs:
            h = conv(batched_graph, h)
            h = F.relu(h)  # Add a non-linear activation
        # Set the updated features back to the graph
        batched_graph.ndata['h'] = h
        # features = h.view(batch_size, node_num, feature_dim)
        # Use dgl.readout_nodes to perform max pooling for each graph in the batch
        features = dgl.readout_nodes(batched_graph, 'h', op='max')
        
        return features