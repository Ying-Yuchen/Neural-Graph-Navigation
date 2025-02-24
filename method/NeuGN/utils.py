
from typing import List, Sequence
from NeuGN.graph_tokenizer import GraphTokenizer
from NeuGN.model import GraphDecoder
import random
import numpy as np
import dgl
import os
import pandas as pd
import csv
import yaml
import argparse
from box import Box
import torch
import torch.distributed as dist
from NeuGN.nx_utils import graph2path_v2

def metrics(res, labels):
    res = np.concatenate(res)
    acc_ar = (res == labels)  # [BS, K]
    acc = acc_ar.sum(-1)

    rank = np.argmax(acc_ar, -1) + 1
    mrr = (acc / rank).mean()
    ndcg = (acc / np.log2(rank + 1)).mean()
    return acc.mean(), mrr, ndcg



def load_ori_graph(path, name):
    node_count = 0
    edge_count = 0
    nodes = []
    node_values = []
    src_ids = []
    dst_ids = []
    file_path = os.path.join(path, f'{name}.graph')
    with open(file_path, "r") as file:
        lines = file.readlines()
        first_line = lines[0].strip().split()
        node_count, edge_count = int(first_line[1]), int(first_line[2])
        
        for line in lines[1:]:
            if line.startswith("v"):
                parts = line.strip().split()
                node_id = int(parts[1]) 
                node_value = int(parts[2]) 
                nodes.append(node_id)
                node_values.append(node_value)
            if line.startswith("e"):
                parts = line.strip().split()
                src_id = int(parts[1])
                dst_id = int(parts[2])
                src_ids.append(src_id)
                dst_ids.append(dst_id)
    
    edge_src_ids = src_ids + dst_ids
    edge_dst_ids = dst_ids + src_ids
    node_values_uni = set(node_values)
    
    return node_values, node_values_uni, edge_src_ids, edge_dst_ids

def load_amazon(path):
    edge_path = os.path.join(path, 'new_edges.csv')
    node_path = os.path.join(path, 'new_nodes.csv')
    edge_data = pd.read_csv(edge_path)
    node_data = pd.read_csv(node_path)
    node_values = node_data['attribute'].tolist()
    node_value_uni = set(node_values)
    src = edge_data['node1_id'].tolist()
    dst = edge_data['node2_id'].tolist()
    edge_src_ids = src + dst
    edge_dst_ids = dst + src
    
    
    return node_values, node_value_uni, edge_src_ids, edge_dst_ids

def save_value2id(value_set, path, name):
    value2id = {str(element): idx for idx, element in enumerate(sorted(value_set))}
    file_path = os.path.join(path, f'{name}_value2id_mapping.csv')
    with open(file_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Element", "ID"]) 
        for element, idx in value2id.items():
            writer.writerow([element, idx]) 
            
    print(f"Mapping has been saved to {file_path}")
    return value2id
    
def load_value2id(path, name=None):
    value2id = {}
    if name == None:
        dataset_list = ['lastfm', 'hamster', 'nell', 'wikics']
        for dname in dataset_list:
            if dname in path:
                name = dname
    file_path = os.path.join(path, f'{name}_value2id_mapping.csv')
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            value2id[line[0]] = int(line[1])
    return value2id
        

def load_model_args(config_path: str):
    with open(os.path.join(config_path, 'model_args.yaml'), 'r') as f:
        args_dict = yaml.safe_load(f)

    return Box(args_dict)
    
def load_checkpoint_single_gpu(filepath, model, device):
    # Load the state dictionary from the checkpoint
    state = torch.load(filepath, map_location=device)  # Assuming you're using GPU with id 0
    # Load the model state dictionary
    model.load_state_dict(state['model_state_dict'])

    print("Model loaded successfully from", filepath)


def load_checkpoint(filepath, model, optimizer, scheduler, rank):
    # Initialize an empty state dictionary
    state = None

    # Create map_location for current rank's GPU
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}  # Assuming rank maps to GPU id
    
    state = torch.load(filepath, map_location=map_location)

    # Load the state into the model, optimizer, and scheduler
    model.module.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    scheduler.load_state_dict(state['scheduler_state_dict'])

    dist.barrier()
    
    return state['epoch']

def save_checkpoint(model, optimizer, scheduler, epoch, filepath):
    if dist.get_rank() == 0:
        state = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),  # 注意：使用 model.module
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 
        }
        torch.save(state, filepath)


def load_model(path, device):
    params = load_model_args(path)
    params.device = device
    encoder_name = params.encoder_config.encoder_name
    
    model = GraphDecoder(params).to(device)
    params_path = os.path.join(path, f'{encoder_name}_checkpoint')
    load_checkpoint_single_gpu(params_path, model, device)
    return model

def replace_and_select(lst, n, x, y):
    unique_numbers = list(set(lst))  
    if n > len(unique_numbers):
        n = len(unique_numbers)
    selected_numbers = random.sample(unique_numbers, n) 
    chosen_number = random.choice(selected_numbers)
    modified_list_ori = [y if item == chosen_number else item for item in lst]
    modified_list = [x if item in selected_numbers else item for item in modified_list_ori]
    return selected_numbers, modified_list, chosen_number


def process_input_data(graph, graph_tokenizer:GraphTokenizer, start_nodes, graph_len, walk_len, epoch, params):

    cur_mask_num = min(int(epoch / 200) + 1, params.dataprocess_config.walk_len)

    if params.dataprocess_config.data_obtain_method == 'random_walk':
        max_len = torch.max(graph_len)
        walks = graph_tokenizer.random_walks(start_nodes, max_len)
        # valid_graphs = []
        valid_graphs = [walks[i, :graph_len[i]].tolist() for i in range(walks.size(0)) if -1 not in walks[i]]
        
    elif params.dataprocess_config.data_obtain_method == 'neighborhood_sampling':
        max_len = 0
        valid_graphs = []
        walks = graph_tokenizer.neighborhoods_sampling(start_nodes, params.dataprocess_config.neighbor_fanouts)
        for walk in walks:
            if -1 not in walk:
                valid_graphs.append(walk)
                if len(walk) > max_len:
                    max_len = len(walk)
        valid_graphs = [walks[i].tolist() for i in range(len(walks)) if -1 not in walks[i]]
            
    sub_graphs = []
    for walk in valid_graphs:
        sub_graphs.append(dgl.node_subgraph(graph, list(set(walk))))
    batch_graphs = dgl.batch(sub_graphs)
    if params.dataprocess_config.data_process_method == 'random_walk':
        valid_walks = [valid_graphs[i][:walk_len[i]] for i in range(len(valid_graphs))]
    elif params.dataprocess_config.data_process_method == 'shuffle':
        valid_walks = []
        for i in range(len(valid_graphs)):
            unique_list = list(set(valid_graphs[i][:walk_len[i]]))
            random.shuffle(unique_list)
            valid_walks.append(unique_list)
    elif params.dataprocess_config.data_process_method == 'one_step':
        valid_walks = [[start_node] for start_node in start_nodes.tolist()]
    elif params.dataprocess_config.data_process_method == 'one_step_random':
        valid_walks = [[random.choice(list(set(valid_graph)))] for valid_graph in valid_graphs]
    elif params.dataprocess_config.data_process_method == 'eular':
        valid_walks = []
        valid_subnodeid_walks = []
        valid_nodeid_results = []
        valid_subnodeid_results = []
        max_len = 0
        for sub_graph in sub_graphs:
            node_id_list = sub_graph.ndata[dgl.NID].tolist()
            start_sub_node_num = random.randint(0, params.decoder_config.sub_node_id_size-1)
            nodeid2subnodeid = {}
            for id in node_id_list:
                nodeid2subnodeid[id] = (start_sub_node_num % params.decoder_config.sub_node_id_size)
                start_sub_node_num += 1
            ###eular
            valid_path_sub = []
            valid_path_tuple = graph2path_v2(sub_graph)
            for edge in valid_path_tuple:
                valid_path_sub.append(edge[0])
            valid_path_sub.append(valid_path_tuple[-1][-1])

            valid_path = [node_id_list[node] for node in valid_path_sub]

            valid_path_subnode = [nodeid2subnodeid[node] for node in valid_path]

            unique_nodes_num = len(list(set(valid_path)))
            replace_num = random.randint(1, min(cur_mask_num, unique_nodes_num))
            selected_numbers, modified_tokens, chosen_number = replace_and_select(valid_path, replace_num, graph_tokenizer.padding_id, graph_tokenizer.sos_id)

            max_len = max(max_len, len(valid_path_subnode))
            valid_walks.append(modified_tokens)
            valid_subnodeid_walks.append(valid_path_subnode)
            valid_nodeid_results.append(chosen_number)
            valid_subnodeid_results.append(nodeid2subnodeid[chosen_number])

            
    valid_tokens_list = valid_walks
    target_token_list = graph_tokenizer.encode_walk(valid_nodeid_results)
    input_tokens_list, input_subnodes_list, token_mask_len_list = [], [], []

    for modified_tokens, subnodeid_walk, subid in zip(valid_tokens_list, valid_subnodeid_walks, valid_subnodeid_results):
        if len(modified_tokens) < max_len:
            padding = [graph_tokenizer.padding_id] * (max_len - len(modified_tokens))
            input_seq = [graph_tokenizer.sos_id] + modified_tokens + padding
            subnode_seq = [subid] + subnodeid_walk + [0] * (max_len - len(modified_tokens))   ###paddding
        else:
            input_seq = [graph_tokenizer.sos_id] + modified_tokens
            subnode_seq = [subid] + subnodeid_walk
        token_mask_len_list.append(len(modified_tokens)+1)
        input_tokens_list.append(input_seq)
        input_subnodes_list.append(subnode_seq)

    input_seqs = torch.tensor(input_tokens_list) 
    input_subnode_seqs = torch.tensor(input_subnodes_list)
    targets = torch.tensor(target_token_list)
    token_mask_len_seq = torch.tensor(token_mask_len_list)
    
        
       
    return batch_graphs, input_seqs, input_subnode_seqs, targets, token_mask_len_seq
