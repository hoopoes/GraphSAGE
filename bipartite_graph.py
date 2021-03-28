# Bipartite Graph Class

import numpy as np
from typing import Dict, Union

# node_id: 'user_0'
# node_index: 0

class Graph(object):
    def __init__(self, node_features: Dict=None, nodes: Dict=None, edges=None):
        """
        Define Base Graph Object to be used in PairSAGE Model.
        This Graph Class is only for bipartite graphs.
        All nodes are represented with their index, not id.

        :param node_features (Dict): dictionary contains node features
         ex) {'user': np.array, 'item': np.array}
        :param nodes (Dict): all nodes
         ex) {'user': [0, 1], 'item': [7001, 7002]}
        :param edges (np.array): link edge ids
         ex) np.array([[0, 7001], [1, 7002]])
        """
        self.node_features = node_features
        self.nodes = nodes
        self.edges = edges

        self.node_type_dict = self.get_node_type_dict()
        self.sampling_index_dict = self.get_sampling_index_dict()
        self.neighbor_dict = self.get_neighbor_dict()

        self.correction_constant = self.node_features['user'].shape[0]

        if not isinstance(node_features, Dict):
            raise TypeError("node features must be python dictionary")

    def __repr__(self):
        identity = "<Graph Object for User-Item Bipartite Graph with {} nodes>".format(
            self.num_nodes)
        return identity

    def __contains__(self, value):
        return value in self.node_type_dict.keys()

    def get_sampling_index_dict(self):
        sampling_index_dict = {node: 0 for node in sorted(list(self.node_type_dict.keys()))}
        return sampling_index_dict

    def get_node_type_dict(self):
        node_type_dict = {user_node: 'user' for user_node in  self.nodes['user']}
        node_type_dict.update({item_node: 'item' for item_node in self.nodes['item']})
        return node_type_dict

    def get_neighbor_dict(self):
        nodes = sorted(list(set(self.edges[0])))
        # edges_index = np.vectorize(self.node_dict.get)(self.edges)
        neighbor_dict = {
            node: self.edges[:, np.where(self.edges[0] == node)[0]][1].tolist()
            for node in nodes
        }
        return neighbor_dict

    def get_neighbors_from_node(self, node: int) -> list:
        neighbors = self.neighbor_dict[node]
        return neighbors

    def get_node_type(self, node: int):
        node_type = self.node_type_dict[node]
        return node_type

    def get_node_features_from_node(self, nodes: list, node_type: str):
        if node_type == 'user':
            node_indices = nodes
        else:
            node_indices = [node_id - self.correction_constant for node_id in nodes]
        features = self.node_features[node_type][node_indices, :]
        return features

    @property
    def num_nodes(self):
        num_nodes = len(self.node_type_dict)
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

