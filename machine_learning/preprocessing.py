import random
import numpy as np
import networkx as nx

def train_data_generator(V = 1000, batch_size = 128, dj = None):
    if dj is None:
        dj_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    else:
        dj_list = [dj]
    
    while True:
        g = nx.barabasi_albert_graph(n=V, m=4)
        for dj_val in dj_list:
            assert batch_size // 2 <= dj_val * g.number_of_edges() # otherwise not enough labeled 1 examples 
            g_mod = hide_edges(g, dj_val)

            # Set for quick edge membership check
            full_edges = set(g.edges())
            observed_edges = set(g_mod.edges())

            # Sample positive edges (label = 1)
            pos_edges = set()
            while len(pos_edges) < batch_size // 2:
                u, v = random.sample(list(g_mod.nodes), 2)
                if (u, v) in observed_edges or (v, u) in observed_edges:
                    continue
                if (u, v) in full_edges or (v, u) in full_edges:
                    pos_edges.add((u, v))
            pos_edges = list(pos_edges)

            # Sample negative edges (non-existent in g)
            neg_edges = set()
            while len(neg_edges) < batch_size // 2:
                u, v = random.sample(list(g_mod.nodes), 2)
                if (u, v) in full_edges or (v, u) in full_edges:
                    continue
                neg_edges.add((u, v))
            neg_edges = list(neg_edges)

            samples = [(u, v, 1) for u, v in pos_edges] + [(u, v, 0) for u, v in neg_edges]
            random.shuffle(samples)

            X = []
            y = []
            bc_map = nx.betweenness_centrality(g_mod, normalized=True)

            for u, v, label in samples:
                features = extract_features(g_mod, u, v, bc_map)
                X.append(features)
                y.append(label)
                
            yield np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    

def extract_features(g:nx.Graph, i:int, j:int, bc_map:dict):
    neighbors_i = set(g.neighbors(i))
    neighbors_j = set(g.neighbors(j))
    common = neighbors_i & neighbors_j
    union = neighbors_i | neighbors_j

    deg_i = g.degree[i]
    deg_j = g.degree[j]
    common_neighbors = len(common)
    jaccard = len(common) / len(union) if union else 0

    bc_i = bc_map.get(i)
    bc_j = bc_map.get(j)

    try:
        shortest_path_length = nx.shortest_path_length(g, source=i, target=j)
    except nx.NetworkXNoPath:
        shortest_path_length = np.inf

    return [deg_i, deg_j, common_neighbors, jaccard, bc_i, bc_j, shortest_path_length]

def hide_edges(g:nx.Graph, dj) -> nx.Graph:
    g_new = g.copy()
    M = int(dj * g_new.number_of_edges())
    edges_list = list(g_new.edges())
    random.shuffle(edges_list)
    g_new.remove_edges_from(edges_list[:M])
    largest_cc = max(nx.connected_components(g_new), key=len)
    g_new = g_new.subgraph(largest_cc).copy()
    return g_new