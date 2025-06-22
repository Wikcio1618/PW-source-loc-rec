import random
import numpy as np
import networkx as nx

def train_data_generator(V=1000, batch_size=128, dj=None):
    if dj is None:
        dj_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    else:
        dj_list = [dj]
    
    while True:
        g = nx.barabasi_albert_graph(n=V, m=4)
        for dj_val in dj_list:
            assert batch_size // 2 <= dj_val * g.number_of_edges()
            g_mod = hide_edges(g, dj_val)

            full_edges = set(map(tuple, map(sorted, g.edges())))
            observed_edges = set(map(tuple, map(sorted, g_mod.edges())))
            node_idxs = list(g_mod.nodes())

            pos_edges = set()
            while len(pos_edges) < batch_size // 2:
                u, v = random.sample(node_idxs, 2)
                if tuple(sorted((u, v))) in observed_edges:
                    continue
                if tuple(sorted((u, v))) in full_edges:
                    pos_edges.add((u, v))
            pos_edges = list(pos_edges)

            neg_edges = set()
            while len(neg_edges) < batch_size // 2:
                u, v = random.sample(node_idxs, 2)
                if tuple(sorted((u, v))) in full_edges:
                    continue
                neg_edges.add((u, v))
            neg_edges = list(neg_edges)

            samples = [(u, v, 1) for u, v in pos_edges] + [(u, v, 0) for u, v in neg_edges]
            random.shuffle(samples)

            pr_map = nx.pagerank(g_mod)
            cc_map = nx.clustering(g_mod)

            X, y = [], []
            for u, v, label in samples:
                features = extract_features(g_mod, u, v, pr_map, cc_map)
                X.append(features)
                y.append(label)

            yield np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    

def extract_features(g: nx.Graph, i: int, j: int, pr_map: dict, cc_map: dict):
    neighbors_i = set(g.neighbors(i))
    neighbors_j = set(g.neighbors(j))
    common = neighbors_i & neighbors_j
    union = neighbors_i | neighbors_j

    deg_i = g.degree[i]
    deg_j = g.degree[j]
    common_neighbors = len(common)
    jaccard = len(common) / len(union) if union else 0.0

    pr_i = pr_map.get(i, 0.0)
    pr_j = pr_map.get(j, 0.0)
    cc_i = cc_map.get(i, 0.0)
    cc_j = cc_map.get(j, 0.0)

    return [deg_i, deg_j, common_neighbors, jaccard, pr_i, pr_j, cc_i, cc_j]

def hide_edges(g:nx.Graph, dj) -> nx.Graph:
    g_new = g.copy()
    M = int(dj * g_new.number_of_edges())
    edges_list = list(g_new.edges())
    random.shuffle(edges_list)
    g_new.remove_edges_from(edges_list[:M])
    largest_cc = max(nx.connected_components(g_new), key=len)
    g_new = g_new.subgraph(largest_cc).copy()
    return g_new