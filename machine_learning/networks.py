import networkx as nx

def build_graph_from_file(path, sep=None):
    G = nx.Graph()
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                u, v = map(int, line.strip().split(sep=sep))
                G.add_edge(u, v)
    return G

def make_EMAIL_graph():
    return build_graph_from_file("../networks/email.txt")

def make_CAL_graph():
    return build_graph_from_file("../networks/california.csv")

def make_FB_graph():
    return build_graph_from_file("../networks/facebook_edges.csv", sep=',')