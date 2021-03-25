# Bipartite Graph Class

import numpy as np
from typing import Dict, Union

# node_id: 'user_0'
# node_index: 0

class Graph(object):
    def __init__(self, node_features: Dict=None, edges=None, node_dict: Dict=None):
        """
        Define Base Graph Object to be used in PairSAGE Model
        This Graph Class is only for bipartite graphs

        :param node_features: Dictionary that contains node features
         ex) {'user': np.array, 'item': np.array}
        :param edges: link edge ids
         ex) np.array([['user_0', 'item_0'], ['item_0', 'user_0']])
        :param node_dict: Dictionary with node_id keys and node_index values
         ex) {'user_0': 0, 'item_0': 1}
        """
        self.node_features = node_features
        self.edges = edges
        self.node_dict = node_dict
        self.node_type_dict_id, self.node_type_dict_index = self.get_node_type_dict()

        self.reverse_node_dict = {val: key for key, val in node_dict.items()}
        self.neighbor_dict_id, self.neighbor_dict_index = self.get_neighbor_dict()
        self.existing_node = list(node_dict.keys())
        self.sampling_index_dict = self.get_sampling_index_dict()

        if not isinstance(node_features, Dict):
            raise TypeError("node features must be python dictionary")

    def __repr__(self):
        identity = "<Graph Object for User-Item Bipartite Graph with {} nodes>".format(
            self.num_nodes)
        return identity

    def __contains__(self, value):
        return value in self.node_dict.keys()

    def get_sampling_index_dict(self):
        sampling_index_dict = {node_id: 0 for node_id in self.node_dict.values()}
        return sampling_index_dict

    def get_node_type_dict(self):
        node_type_dict_id = {key: key.split('_')[0] for key in self.node_dict.keys()}
        node_type_dict_index = {self.node_dict[key]: key.split('_')[0] for key in self.node_dict.keys()}
        return node_type_dict_id, node_type_dict_index

    def get_neighbor_dict(self):
        node_ids = sorted(list(set(self.edges[0])))
        neighbor_dict_id = {
            node_id: self.edges[:, np.where(self.edges[0] == node_id)[0]][1].tolist()
            for node_id in node_ids
        }

        edges_index = np.vectorize(self.node_dict.get)(self.edges)
        node_indices = sorted(list(set(edges_index[0])))
        neighbor_dict_index = {
            node_index: edges_index[:, np.where(edges_index[0] == node_index)[0]][1].tolist()
            for node_index in node_indices
        }
        return neighbor_dict_id, neighbor_dict_index

    def get_neighbors_from_node(self, node: Union[int, str]) -> list:
        if type(node) == int:
            neighbors = self.neighbor_dict_index[node]
        else:
            neighbors = self.neighbor_dict_id[node]
        return neighbors

    def get_node_type(self, node: Union[int, str]):
        if type(node) == int:
            node = self.reverse_node_dict[node]
            node_type = self.node_type_dict_id[node]
        else:
            node_type = self.node_type_dict_id[node]
        return node_type

    def get_node_features_from_node(self, nodes: list, node_type: str):
        correction_constant = self.node_features['user'].shape[0]

        if type(nodes[0]) == int:
            if node_type == 'user':
                node_indices = nodes
            else:
                node_indices = [node_id - correction_constant for node_id in nodes]
        else:
            # node_ids = ['user_0', 'user_1'], node_type = 'user'
            if node_type == 'user':
                node_indices = self.change_id_to_index(nodes)
            else:
                node_indices = [self.node_dict[node_id] - correction_constant for node_id in nodes]
        features = self.node_features[node_type][node_indices, :]
        return features

    def change_id_to_index(self, node_ids):
        node_indices = [self.node_dict[node_id] for node_id in node_ids]
        return node_indices

    @property
    def num_nodes(self):
        num_nodes = len(self.node_dict)
        return num_nodes

    @property
    def num_edges(self):
        num_edges = self.edges.shape[1]
        return num_edges

    @property
    def node_feature_sizes(self):
        num_user_features = self.node_features['user'].shape[1]
        num_item_features = self.node_features['item'].shape[1]
        node_feature_sizes = {'user': num_user_features, 'item': num_item_features}
        return node_feature_sizes

