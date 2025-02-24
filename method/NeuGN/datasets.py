from llama.graph_tokenizer import GraphTokenizer
import torch
from torch.utils.data import Dataset
import random

class GraphWalkDataset(Dataset):
    def __init__(self, graph_tokenizer: GraphTokenizer, max_len: int, mode: str, rand_pre: bool):
        self.tokenizer = graph_tokenizer
        self.mode = mode
        self.rand_pre = rand_pre
        if mode == 'train':
            self.num_walks = int(self.tokenizer.node_nums() * 0.8)
            self.bias = 0
        elif mode == 'trans_test' :
            self.num_walks = self.tokenizer.node_nums() -  int(self.tokenizer.node_nums() * 0.8)
            self.bias = self.tokenizer.node_nums() * 0.8
        elif mode == 'all':
            self.num_walks = int(self.tokenizer.node_nums())
            self.bias = 0
        
        self.max_len = max_len

    def __len__(self):
        return self.num_walks

    def __getitem__(self, idx):
        start_node = idx + self.bias
        graph_len = random.randint(2, self.max_len+1)
        walk_len = random.randint(1, graph_len)
        
        return torch.tensor(start_node), torch.tensor(graph_len), torch.tensor(walk_len)