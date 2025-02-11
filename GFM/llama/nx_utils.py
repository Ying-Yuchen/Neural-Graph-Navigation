import sys
import random
import networkx as nx
import itertools
from typing import List, Tuple, Dict, Union, Iterable
import torch
from torch import Tensor
from dgl import to_networkx, DGLGraph
import dgl


def graph2path_v2(graph: DGLGraph) -> List[Tuple[int, int]]:
    G = to_networkx(graph).to_undirected()
    # 1. create list of subgraphs
    if not nx.is_connected(G):
        S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    else:
        S = [G]
    # 2. find eulerian paths in each subgraph, and then concat sub-paths
    random.shuffle(S)
    s = S[0]
    path = connected_graph2path(s)
    prev_connect_node = list(s.nodes)[0] if len(path) == 0 else path[-1][-1]
    for s in S[1:]:
        spath = connected_graph2path(s)
        if len(spath) == 0:  # single node
            curr_connect_node = list(s.nodes)[0]
        else:
            curr_connect_node = spath[0][0]
        jump_edge = (prev_connect_node, curr_connect_node)
        path.append(jump_edge)
        path.extend(spath)
        prev_connect_node = path[-1][-1]
    return path

def connected_graph2path(G) -> List[Tuple[int, int]]:
    if len(G.nodes) == 1:
        path = []
    else:
        if not nx.is_eulerian(G):
            G = nx.eulerize(G)
        node = random.choice(list(G.nodes()))
        raw_path = list(_customized_eulerian_path(G, source=node))
        path = shorten_path(raw_path)
    return path

def _customized_eulerian_path(G, source):
    # To enhance randomization of eulerian path, thus as a kind of data augmentation
    if random.random() < 0.5:
        return nx.eulerian_path(G, source=source)
    else:
        return nx.eulerian_circuit(G, source=source)
    
def shorten_path(path):
    """
    If the given path is euler path, then it will go back to the start node, meaning that some edges are duplicated after
    all edges have been visited. So we need to remove those unnecessary edges.
    If the given path is semi-euler path, then usually there is no unnecessarily repeated edges.
    :param path:
    :return:
    """
    triangle_path = [(src, tgt) if src < tgt else (tgt, src) for src, tgt in path]
    unique_edges = set(triangle_path)
    idx = 0
    for i in range(1, len(path) + 1):
        short_path = triangle_path[:i]
        if set(short_path) == unique_edges:
            idx = i
            break
    path = path[:idx]
    return path


