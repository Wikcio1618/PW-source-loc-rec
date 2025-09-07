from networks import make_FB_graph

import random
import numpy as np
import networkx as nx

def train_data_generator(batch_size=64, dj=None):
    g = make_FB_graph()
    while True:
        # g = nx.barabasi_albert_graph(int(random.randint(650, 1400)), m=random.randint(1, 5))
        dj = random.random() * 0.6
        g_mod, hidden_edges = hide_edges(g, dj)
        assert batch_size // 2 <= len(hidden_edges)

        full_edges = set(map(tuple, map(sorted, g.edges())))
        node_idxs = list(g_mod.nodes())

        pos_edges = set()
        i = 0
        while len(pos_edges) < batch_size // 2:
            u, v = hidden_edges[i]
            if g_mod.has_node(u) and g_mod.has_node(v):
                pos_edges.add(hidden_edges[i])
            i += 1
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

        # Precompute node-level / global features
        bc_map = nx.betweenness_centrality(g_mod)
        pl_map = dict(nx.shortest_path_length(g_mod, weight=None))
        pr_map = nx.pagerank(g_mod)
        cc_map = nx.clustering(g_mod)

        # num_edges = g_mod.number_of_edges()
        mean_degree = sum(dict(g_mod.degree()).values()) / g_mod.number_of_nodes()
        density = nx.density(g_mod)

        X, y = [], []
        for u, v, label in samples:
            features = extract_features(
                g_mod, u, v,
                pr_map, cc_map, bc_map, pl_map,
                mean_degree, density
            )
            X.append(features)
            y.append(label)

        yield np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def extract_features(g: nx.Graph, i: int, j: int,
                     pr_map: dict, cc_map: dict, bc_map: dict, pl_map: dict, 
                     mean_degree: float, density: float):
    # Local similarity scores
    jaccard = next(nx.jaccard_coefficient(g, [(i, j)]))[2]
    adamic_adar = next(nx.adamic_adar_index(g, [(i, j)]))[2]
    resource_alloc = next(nx.resource_allocation_index(g, [(i, j)]))[2]
    pref_attach = next(nx.preferential_attachment(g, [(i, j)]))[2]

    # Node-level
    bc_i = bc_map.get(i, 0.0)
    bc_j = bc_map.get(j, 0.0)
    pr_i = pr_map.get(i, 0.0)
    pr_j = pr_map.get(j, 0.0)
    cc_i = cc_map.get(i, 0.0)
    cc_j = cc_map.get(j, 0.0)

    # Distance
    path_length = pl_map[i].get(j, 0.0)

    return [
        jaccard, adamic_adar, resource_alloc, pref_attach,
        bc_i, bc_j, pr_i, pr_j, cc_i, cc_j,
        path_length, mean_degree, density
    ]

def hide_edges(g:nx.Graph, dj) -> nx.Graph:
    g_new = g.copy()
    M = int(dj * g_new.number_of_edges())
    edges_list = list(g_new.edges())
    random.shuffle(edges_list)
    g_new.remove_edges_from(edges_list[:M])
    largest_cc = max(nx.connected_components(g_new), key=len)
    g_new = g_new.subgraph(largest_cc).copy()
    return g_new, edges_list