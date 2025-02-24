import torch
import os
from datetime import datetime
os.environ['DGLBACKEND'] = 'pytorch'
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from NeuGN.graph_tokenizer import GraphTokenizer
from NeuGN.model import Transformer, GraphDecoder
from NeuGN.utils import metrics, load_amazon, load_ori_graph, save_value2id, load_value2id, load_model_args, load_checkpoint, save_checkpoint, process_input_data
from NeuGN.datasets import GraphWalkDataset
import ipdb
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import yaml
from dataclasses import dataclass, asdict

import dgl
import time
import argparse
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
# from transformers import Trainer
from NeuGN.nx_utils import graph2path_v2
import random


def train_one_epoch(model, graph, graph_tokenizer, dataloader, criterion, optimizer, params, device, epoch, writer):
    model.train()
    # model.eval()
    total_loss = 0
    dataloader.sampler.set_epoch(epoch)
    res100, res50, res20, res10, res1, labels = [], [], [], [], [], []
    hit1_dict, hit10_dict, hit20_dict, hit50_dict, hit100_dict, mrr1_dict, mrr10_dict, mrr20_dict, mrr50_dict, mrr100_dict = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    # res50 = []
    progress_bar = tqdm(dataloader, desc="Training")

    for start_nodes, graph_len, walk_len in progress_bar:
        batch_graphs, input_seqs, input_subnode_seqs, targets, token_mask_len_seq = process_input_data(graph, graph_tokenizer, start_nodes, graph_len, walk_len, epoch, params)

        batch_graphs, input_seqs, input_subnode_seqs, targets, token_mask_len_seq = batch_graphs.to(device), input_seqs.to(device),  input_subnode_seqs.to(device), targets.to(device), token_mask_len_seq.to(device)
        
        if params.decoder_config.decoder_type == 'llama':
            outputs = model(batch_graphs, input_seqs, input_subnode_seqs, token_mask_len_seq, 0, device)[:, 0]
        elif params.decoder_config.decoder_type == 'mlp':
            outputs = model(batch_graphs, input_seqs, 0, device)

        res100.append(outputs.topk(100)[1].cpu())
        res50.append(outputs.topk(50)[1].cpu())
        res20.append(outputs.topk(20)[1].cpu())
        res10.append(outputs.topk(10)[1].cpu())
        res1.append(outputs.topk(1)[1].cpu())
        labels.append(targets.cpu())

        optimizer.zero_grad()
        
        loss = criterion(outputs, targets)
   
        loss.backward()
        
        optimizer.step()

        total_loss += loss.item()
        
        progress_bar.set_postfix(batch_loss=loss.item()/input_seqs.size(0))

    avg_loss = total_loss / len(dataloader)
    labels = np.concatenate(labels)
    labels = labels.reshape(-1, 1)
    acc100, mrr100, ndcg100 = metrics(res100, labels)
    acc50, mrr50, ndcg50 = metrics(res50, labels)
    acc20, mrr20, ndcg20 = metrics(res20, labels)
    acc10, mrr10, ndcg10 = metrics(res10, labels)
    acc1, mrr1, ndcg1 = metrics(res1, labels)
    print(f'mrr1:{mrr1*100}, mrr10:{mrr10*100}, mrr20:{mrr20*100}, mrr50:{mrr50*100}, mrr100:{mrr100*100}')
    print(f'acc1:{acc1*100}, acc10:{acc10*100}, acc20:{acc20*100}, acc50:{acc50*100}, acc100:{acc100*100}')
    print(f"Average Loss: {avg_loss}")
    hit1_dict[f'train_acc1'] = acc1*100
    hit10_dict[f'train_acc10'] = acc10*100
    hit20_dict[f'train_acc20'] = acc20*100
    hit50_dict[f'train_acc50'] = acc50*100
    hit100_dict[f'train_acc100'] = acc100*100
    mrr1_dict[f'train_mrr1'] = mrr1*100
    mrr10_dict[f'train_mrr10'] = mrr10*100
    mrr20_dict[f'train_mrr20'] = mrr20*100
    mrr50_dict[f'train_mrr50'] = mrr50*100
    mrr100_dict[f'train_mrr100'] = mrr100*100
    writer.add_scalar(f'train_average_loss', avg_loss, epoch)
    writer.add_scalars(f'train_hit1', hit1_dict, epoch)
    writer.add_scalars(f'train_hit10', hit10_dict, epoch)
    writer.add_scalars(f'train_hit20', hit20_dict, epoch)
    writer.add_scalars(f'train_hit50', hit50_dict, epoch)
    writer.add_scalars(f'train_hit100', hit100_dict, epoch)
    writer.add_scalars(f'train_mrr1', mrr1_dict, epoch)
    writer.add_scalars(f'train_mrr10', mrr10_dict, epoch)
    writer.add_scalars(f'train_mrr20', mrr20_dict, epoch)
    writer.add_scalars(f'train_mrr50', mrr50_dict, epoch)
    writer.add_scalars(f'train_mrr100', mrr100_dict, epoch)
    return avg_loss

def eval_one_epoch(model, dataloader1, dataloader20, dataloader_rand, criterion, device, epoch, test_data_num, writer):
    model.eval()
    hit1_dict = {}
    hit10_dict = {}
    hit20_dict = {}
    mrr1_dict = {}
    mrr10_dict = {}
    mrr20_dict = {}

    dataloader_list = [dataloader1, dataloader20, dataloader_rand]
    dataloader_name = ['last', 'normal', 'random']
    for dataloader, name in zip(dataloader_list, dataloader_name):
        total_loss = 0
        res20 = []
        res10 = []
        res1 = []
        labels = []

    
        progress_bar = tqdm(dataloader, desc=f"{name}_eval")
        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs, 0)[:, -1]

                res20.append(outputs.topk(20)[1].cpu())
                res10.append(outputs.topk(10)[1].cpu())
                res1.append(outputs.topk(1)[1].cpu())
                labels.append(targets.cpu())

                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1)) / inputs.size(0)

                total_loss += loss.item()

                progress_bar.set_postfix(batch_loss=loss.item()/inputs.size(0))

        avg_loss = total_loss / test_data_num
        labels = np.concatenate(labels)
        labels = labels.reshape(-1, 1)
        acc20, mrr20, ndcg20 = metrics(res20, labels)
        acc10, mrr10, ndcg10 = metrics(res10, labels)
        acc1, mrr1, ndcg1 = metrics(res1, labels)
        print(f'{name}: test mrr1:{mrr1*100}, mrr10:{mrr10*100}, mrr20:{mrr20*100}')
        print(f'{name}: test acc1:{acc1*100}, acc10:{acc10*100}, acc20:{acc20*100}')
        print(f"{name}: test Average Loss: {avg_loss}")
        hit1_dict[f'{name}_acc1'] = acc1*100
        hit10_dict[f'{name}_acc10'] = acc10*100
        hit20_dict[f'{name}_acc20'] = acc20*100
        mrr1_dict[f'{name}_mrr1'] = mrr1*100
        mrr10_dict[f'{name}_mrr10'] = mrr10*100
        mrr20_dict[f'{name}_mrr20'] = mrr20*100
        writer.add_scalar(f'{name}_average_loss', avg_loss, epoch)
    writer.add_scalars(f'hit1', hit1_dict, epoch)
    writer.add_scalars(f'hit10', hit10_dict, epoch)
    writer.add_scalars(f'hit20', hit20_dict, epoch)
    writer.add_scalars(f'mrr1', mrr1_dict, epoch)
    writer.add_scalars(f'mrr10', mrr10_dict, epoch)
    writer.add_scalars(f'mrr20', mrr20_dict, epoch)
    

    return avg_loss

def main(args):
    # Device configuration
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend='nccl')
        
    params = load_model_args(args.config_path)
    
    print(params.encoder_config)
    
    dataset_list = [ 'lastfm', 'hamster', 'nell', 'wikics']
    dataset_name = None
    for name in dataset_list:
        if name in params.graph_path:
            dataset_name = name
            if dataset_name == 'amazon_ori':
                node_values, node_values_uni, edge_src_ids, edge_dst_ids = load_amazon(params.graph_path)
            else:
                node_values, node_values_uni, edge_src_ids, edge_dst_ids = load_ori_graph(params.graph_path, dataset_name)
        

    now = datetime.now()
    formatted_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    writer_path = f'../../../experiments/results/logs_{params.encoder_config.encoder_name}_{params.decoder_config.decoder_type}_{formatted_time}'
    writer = SummaryWriter(writer_path)
    with open(os.path.join(writer_path, 'model_args.yaml'), 'w') as f:
        yaml.dump(params.to_dict(), f, default_flow_style=False, sort_keys=False)
        
    value2id = save_value2id(node_values_uni, args.config_path, dataset_name)
    
    value_nums = len(node_values_uni)
    num_nodes = len(node_values)
    edge_index_src = torch.tensor([edge_src_ids], dtype=torch.long).squeeze(0)
    edge_index_dst = torch.tensor([edge_dst_ids], dtype=torch.long).squeeze(0)
    
    graph = dgl.graph((edge_index_src, edge_index_dst), num_nodes=num_nodes)
    node_values_id = torch.tensor([value2id[str(node_value)] for node_value in node_values])
    print(len(node_values_id))
    graph.ndata['feat_id'] = node_values_id
    


    # Load data and tokenizer
    tokenizer = GraphTokenizer(graph=graph)
    # params = ModelArgs()
    
    params.decoder_config.vocab_size = tokenizer.token_nums()
    params.encoder_config.graph_value_num = value_nums
    print(params.decoder_config.vocab_size)

    with open(os.path.join(args.config_path, 'model_args.yaml'), 'w') as f:
        yaml.dump(params.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    params.device = device
    # Initialize model
    model = GraphDecoder(params).to(device)
    print(0)
    # Create datasets
    train_dataset = GraphWalkDataset(tokenizer, max_len=params.dataprocess_config.walk_len, mode='all', rand_pre=False)
    
    # Use DistributedSampler to split data across GPUs
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=4)

    model = DDP(model,device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Optimizer and loss function
    # optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.9))
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.padding_id)
    checkpoint_file = os.path.join(params.checkpoint_path, f'{params.encoder_config.encoder_name}_checkpoint')

    # Train model
    num_epochs = args.epochs
    start_epoch = 0
    if args.load_params:
        start_epoch = load_checkpoint(checkpoint_file, model, optimizer, scheduler, local_rank)
    
    
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_one_epoch(model, graph, tokenizer, train_dataloader, criterion, optimizer, params, device, epoch, writer)
        scheduler.step()
        if epoch % 20 ==0 and dist.get_rank() == 0 and epoch != 0:
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_file)
        

parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", default=-1, type=int)
parser.add_argument("--epochs", default=5000, type=int)
parser.add_argument('--load_params', default=1, type=int)
parser.add_argument('--config_path', default='./model_params/wikics', type=str)

args = parser.parse_args()
if __name__ == "__main__":
    main(args)
