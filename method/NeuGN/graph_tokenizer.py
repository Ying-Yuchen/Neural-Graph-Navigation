import torch
from dgl.sampling import random_walk
from dgl.dataloading import NeighborSampler
import dgl
import random

# from torch_geometric.utils import random_walk
import pandas as pd
from typing import List, Sequence

class GraphTokenizer:
    """
    Tokenizer for graph nodes using PyTorch Geometric and random walk sequences.
    """

    def __init__(self, graph: dgl.DGLGraph):
        self.graph = graph
        
        # Get the number of nodes
        self.node_num = self.graph.num_nodes()
        
        self.padding_id = self.node_num
        self.sos_id = self.padding_id+1

        # Initialize tokenizer mappings
        self.node_to_token = {node_id: idx for idx, node_id in enumerate(range(self.node_num))}
        self.node_to_token[self.padding_id] = self.padding_id
        
        self.token_to_node = {idx: node_id for node_id, idx in self.node_to_token.items()}

    def node_nums(self):
        return self.node_num

    def token_nums(self):
        return self.node_num + 2

    def random_walk(self, start_node: int, length: int) -> List[int]:
        """
        Perform random walks from the start nodes.

        Args:
            start_nodes (Sequence[int]): The starting nodes for the random walk.
            length (int): The length of the walk.

        Returns:
            List[int]: A random walk sequence.
        """
        # print(start)
        start = torch.tensor([start_node], dtype=torch.long)
        walk = random_walk(self.graph.edge_index[0], self.graph.edge_index[1], start=start, walk_length=length)
        return walk.tolist()[0]
    
    def neighborhoods_sampling(self, start_nodes, fanouts_max):
        seed_nodes_list = []
        for node in start_nodes:
            if self.graph.in_degrees(node) == 0:
                continue
            out_len = random.randint(1, len(fanouts_max))
            fanouts = []
            for i in range(out_len):
                out = random.randint(1, fanouts_max[i])
                fanouts.append(out)
            sampler = NeighborSampler(fanouts)
            seed_nodes, _, _ = sampler.sample_blocks(self.graph, node)
            seed_nodes_list.append(seed_nodes)
        return seed_nodes_list

    def random_walks(self, start_nodes: Sequence[int], length: int) -> List[List[int]]:
        """
        Perform random walks from the start nodes.

        Args:
            start_nodes (Sequence[int]): The starting nodes for the random walk.
            length (int): The length of the walk.

        Returns:
            List[List[int]]: A list of random walk sequences.
        """
        start = torch.tensor(start_nodes, dtype=torch.long)
        walks, _ = random_walk(self.graph, nodes=start, length=length)
        return walks

    def encode_walks(self, walks: List[List[int]]) -> List[List[int]]:
        """
        Encode random walk sequences into token IDs.

        Args:
            walks (List[List[int]]): The random walk sequences.

        Returns:
            List[List[int]]: The encoded sequences as token IDs.
        """
        return [[self.node_to_token[node] for node in walk] for walk in walks]

    def encode_walk(self, walk: List[int]) -> List[int]:
        """
        Encode random walk sequences into token IDs.

        Args:
            walk (List[int]): The random walk sequences.

        Returns:
            List[int]: The encoded sequences as token IDs.
        """
        return [self.node_to_token[node] for node in walk]

    def decode_walks(self, token_sequences: List[List[int]]) -> List[List[int]]:
        """
        Decode token ID sequences back to node IDs.

        Args:
            token_sequences (List[List[int]]): The token ID sequences.

        Returns:
            List[List[int]]: The decoded sequences as node IDs.
        """
        return [[self.token_to_node[token] for token in sequence] for sequence in token_sequences]


