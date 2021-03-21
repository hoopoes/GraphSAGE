# Experiments
import numpy as np
import pandas as pd
from typing import Dict

# from stellargraph.core.indexed_array import IndexedArray
# torch_geometric.data.Data

users =np.array([[-1], [4]])
items = np.array([[0.4, 100], [0.1, 200], [0.9, 300]])
nodes = {'users': users, 'items': items}
edges = np.array([[0, 0, 0, 1, 1, 2, 3, 3, 4, 4], [2, 3, 4, 3, 4, 0, 0, 1, 0, 1]])
node_dict = {'user_0': 0, 'user_1': 1, 'item_1': 2, 'item_2': 3, 'item_3': 4}

class Graph(object):
    def __init__(self, nodes: Dict=None, edges=None, node_dict=None):
        self.nodes = nodes    # dict
        self.edges = edges
        self.node_dict = node_dict

        self.neighbor_dict  = self.get_neighbor_dict()

        if not isinstance(nodes, Dict):
            raise TypeError("nodes must be python dictionary")

    def __repr__(self):
        identity = "<Graph Object for User-Item Bipartite Graph with {} nodes>".format(
            self.num_nodes)
        return identity

    def __contains__(self, value):
        return value in node_dict.keys()

    def get_neighbor_dict(self):
        node_list = sorted(list(set(self.edges[0])))
        neighbor_dict = {
            node_id: edges[:, np.where(edges == node_id)[1]][1].tolist()
            for node_id in node_list
        }
        return neighbor_dict

    def get_neighbors_from_node(self, node):
        neighbors = self.neighbor_dict[node]
        return neighbors

    @property
    def num_nodes(self):
        num_nodes = len(self.node_dict)
        return num_nodes

    @property
    def num_edges(self):
        num_edges = self.edges.shape[0]
        return num_edges

    @property
    def num_user_features(self):
        num_user_features = self.nodes['user'].shape[1]
        return num_user_features

    @property
    def num_item_featfures(self):
        num_item_features = self.nodes['item'].shape[1]
        return num_item_features


graph = Graph(nodes=nodes, edges=edges, node_dict=node_dict)



