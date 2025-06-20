import torch
import networkx as nx

def extract_features(g:nx.Graph, i:int, j:int, bc_map):
    neighbors_i = set(g.neighbors(i))
    neighbors_j = set(g.neighbors(j))
    common = neighbors_i & neighbors_j
    union = neighbors_i | neighbors_j

    deg_i = g.degree[i]
    deg_j = g.degree[j]
    common_neighbors = len(common)
    jaccard = len(common) / len(union) if union else 0
    pref_attach = deg_i * deg_j
    ra_index = sum(1 / g.degree[z] for z in common) if common else 0

    return [deg_i, deg_j, common_neighbors, jaccard, pref_attach, ra_index]