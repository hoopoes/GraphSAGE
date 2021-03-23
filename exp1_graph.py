# Experiments
import numpy as np
import pandas as pd
from typing import Dict, Union
from collections import Iterable

from stellargraph.core.indexed_array import IndexedArray
# torch_geometric.data.Data

# node_id: 'user_0'
# node_index: 0

users = np.array([[-1], [4]])
items = np.array([[0.4, 100], [0.1, 200], [0.9, 300]])
node_features = {'user': users, 'item': items}
#edges = np.array([[0, 0, 0, 1, 1, 2, 3, 3, 4, 4],
#                  [2, 3, 4, 3, 4, 0, 0, 1, 0, 1]])
edges = np.array([['user_0', 'user_0', 'user_0', 'user_1', 'user_1',
                   'item_0', 'item_1', 'item_2', 'item_1', 'item_2'],
                  ['item_0', 'item_1', 'item_2', 'item_1', 'item_2',
                   'user_0', 'user_0', 'user_0', 'user_1', 'user_1']])
node_dict = {'user_0': 0, 'user_1': 1, 'item_0': 2, 'item_1': 3, 'item_2': 4}

df = pd.DataFrame(edges.T).rename({0: 'user', 1: 'item'}, axis=1).head(5)
df['score'] = [1, 0, 0, 1, 0]
train = list(df[["user", "item"]].itertuples(index=False))

class Graph(object):
    def __init__(self, node_features: Dict=None, edges=None, node_dict=None):
        self.node_features = node_features    # dict
        self.edges = edges
        self.node_dict = node_dict
        self.reverse_node_dict = {val: key for key, val in node_dict.items()}

        self.neighbor_dict  = self.get_neighbor_dict()

        if not isinstance(node_features, Dict):
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
            node_id: edges[:, np.where(edges[0] == node_id)[0]][1].tolist()
            for node_id in node_list
        }
        return neighbor_dict

    def get_neighbors_from_node(self, node: Union[int, str]) -> list:
        if type(node) == int:
            node = self.reverse_node_dict[node] # node: 'user_0'
        neighbors = self.neighbor_dict[node]
        return neighbors

    def get_node_type(self, nodes: Union[int, str]) -> list:
        # 기본 Input: ['user_1']
        if not isinstance(nodes, list):
            nodes = [nodes]
        if type(nodes[0]) == str:
            self.check_node_existence(nodes)
            node_types = [node.split('_')[0] for node in nodes]
        else:
            nodes = [self.reverse_node_dict[node] for node in nodes]
            self.check_node_existence(nodes)
            node_types = [node.split('_')[0] for node in nodes]
        return node_types

    def check_node_existence(self, nodes: list):
        existing_nodes = [node for node in nodes if node in list(self.node_dict.keys())]
        if len(nodes) != len(existing_nodes):
            raise KeyError("Some of nodes do not exist")

    def get_node_features_from_node(self, node_ids: list, node_type: str):
        # node_ids = ['user_0', 'user_1'], node_type = 'user'
        if node_type == 'user':
            node_indices = [self.node_dict[node_id] for node_id in node_ids]
        else:
            correction_constant = self.node_features['user'].shape[0]
            node_indices = [self.node_dict[node_id] - correction_constant for node_id in node_ids]
        features = self.node_features[node_type][node_indices, :]
        return features

    @property
    def num_nodes(self):
        num_nodes = len(self.node_dict)
        return num_nodes

    @property
    def num_edges(self):
        num_edges = self.edges.shape[1]
        return num_edges

    @property
    def num_user_features(self):
        num_user_features = self.node_features['user'].shape[1]
        return num_user_features

    @property
    def num_item_featfures(self):
        num_item_features = self.node_features['item'].shape[1]
        return num_item_features


graph = Graph(node_features=node_features, edges=edges, node_dict=node_dict)


