# LinkSequence, BatchedLinkGenerator, PairSAGEGenerator

import collections
from collections.abc import Iterable
import numpy as np
from tensorflow.keras.utils import Sequence

from utils import is_real_iterable, random_state

# Sequence
# Base object for fitting to a sequence of data, such as a dataset.

class LinkSequence(Sequence):
    """
    Keras-compatible data generator to use with Keras methods
    :meth:`keras.Model.fit/evaluate/predict`.

    Args:
        sample_function (Callable): A function that returns features for supplied head nodes.
        batch_size
        ids (iterable): Link IDs to batch, each link id being a tuple of (source, destination) node ids.
        targets (list, optional): A list of targets or labels to be used in the downstream task.
        shuffle (bool): If True (default) the ids will be randomly shuffled every epoch.
        seed (int, optional): Random seed
    """

    def __init__(
        self, sample_function, batch_size, ids, targets=None, shuffle=True, seed=None
    ):
        # Check that ids is an iterable
        if not is_real_iterable(ids):
            raise TypeError("IDs must be an iterable or numpy array of graph node IDs")

        # Check targets is iterable & has the correct length
        if targets is not None:
            if not is_real_iterable(targets):
                raise TypeError("Targets must be None or an iterable or numpy array ")
            if len(ids) != len(targets):
                raise ValueError(
                    "The length of the targets must be the same as the length of the ids")
            self.targets = np.asanyarray(targets)
        else:
            self.targets = None

        # Ensure number of labels matches number of ids
        if targets is not None and len(ids) != len(targets):
            raise ValueError("Length of link ids must match length of link targets")

        # Store the generator to draw samples from graph
        if isinstance(sample_function, collections.abc.Callable):
            self._sample_features = sample_function
        else:
            raise TypeError(
                "({}) The sampling function expects a callable function.".format(
                    type(self).__name__))

        self.batch_size = batch_size
        self.ids = list(ids)
        self.data_size = len(self.ids)
        self.shuffle = shuffle
        self._rs, _ = random_state(seed)

        # Shuffle the IDs to begin
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self, batch_num):
        """
        Generate one batch of data
        Args:
            batch_num (int): number of a batch
        Returns:
            batch_feats (list): Node features for nodes and neighbours sampled from a
                batch of the supplied IDs
            batch_targets (list): Targets/labels for the batch.
        """
        start_idx = self.batch_size * batch_num
        end_idx = start_idx + self.batch_size

        if start_idx >= self.data_size:
            raise IndexError("Mapper: batch_num larger than length of data")
        # print("Fetching {} batch {} [{}]".format(self.name, batch_num, start_idx))

        # The ID indices for this batch
        batch_indices = self.indices[start_idx:end_idx]

        # Get head (root) nodes for links
        head_ids = [self.ids[ii] for ii in batch_indices]

        # Get targets for nodes
        batch_targets = None if self.targets is None else self.targets[batch_indices]

        # Get node features for batch of link ids
        batch_feats = self._sample_features(head_ids, batch_num)

        return batch_feats, batch_targets

    def on_epoch_end(self):
        # Shuffle all link IDs at the end of each epoch
        self.indices = list(range(self.data_size))
        if self.shuffle:
            self._rs.shuffle(self.indices)

# ----------
# Generator
# stellagraph, pairsage generator

import abc

class Generator(abc.ABC):
    @abc.abstractmethod
    def num_batch_dims(self):
        # Returns the number of batch dimensions in returned tensors (_not_ the batch size itself).
        # Ex) feature shape: (1, num_nodes, feature_size)
        pass

    @abc.abstractmethod
    def flow(self, *args, **kwargs):
        # Create a Tensorflow Keras Sequence for GNN
        pass


from sampler import BreadthFirstWalker

import operator
import collections
from functools import reduce


class BatchedLinkGenerator(Generator):
    def __init__(self, Graph, batch_size):
        self.graph = Graph
        self.batch_size = batch_size
        self.multiplicity = 2    # requires a model with 2 root nodes per query
        self.head_node_types = None
        self.sampler = None

    @abc.abstractmethod
    def sample_features(self, head_links, batch_num):
        pass

    def num_batch_dims(self):
        return 1

    def flow(self, link_ids, targets=None, shuffle=False, seed=None):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        The node IDs are the nodes to train or inference on: the embeddings
        calculated for these nodes are passed to the downstream task. These
        are a subset of the nodes in the graph.

        shuffle argument should be True for training and False for prediction.

        Args:
            link_ids: an iterable of tuples of node IDs as (source, target)
            targets: a 2D array of numeric targets with shape
                ``(len(link_ids), target_size)``
            shuffle (bool): If True the links will be shuffled at each
                epoch, if False the links will be processed in order.
            seed (int, optional): Random seed
        """
        # source -> destination
        if self.head_node_types is not None:
            expected_src_type = self.head_node_types[0]
            expected_dst_type = self.head_node_types[1]

        # pass iterable to standard LinkSequence
        # elif isinstance(link_ids, collections.abc.Iterable):
        # Check all IDs are actually in the graph and are of expected type
        for link in link_ids:
            if len(link) != 2:
                raise KeyError("Expected link IDs to be a tuple of length 2")

            # node 잘 있는지 확인
            src, dst = link
            try:
                node_type_src = self.graph.get_node_type(src)
            except KeyError:
                raise KeyError(
                    f"Node ID {src} supplied to generator not found in graph")
            try:
                node_type_dst = self.graph.get_node_type(dst)
            except KeyError:
                raise KeyError(
                    f"Node ID {dst} supplied to generator not found in graph"
                )

            if self.head_node_types is not None and (
                node_type_src != expected_src_type
                or node_type_dst != expected_dst_type
            ):
                raise ValueError(
                    f"Node pair ({src}, {dst}) not of expected type ({expected_src_type}, {expected_dst_type})"
                )

            # Pandas(user_id='user_0', movie_id='item_3') -> array([0, 41]) 로 변형
            # link_ids = [self.graph.node_ids_to_ilocs(ids) for ids in link_ids]
            link_ids = [self.graph.change_id_to_index([id for id in link_id]) for link_id in link_ids]
            # [[0, 2], [0, 3], [0, 4], [1, 3], [1, 4]]

            return LinkSequence(
                self.sample_features,
                self.batch_size,
                link_ids,
                targets=targets,
                shuffle=shuffle,
                seed=seed)



class PairSAGEGenerator(BatchedLinkGenerator):
    def __init__(
        self,
        Graph,
        batch_size,
        num_samples,
        head_node_types=None,
        seed=None
    ):
        """
        PairSAGE Batch Link Generator Object

        Use the `flow` method supplying the nodes and (optionally) targets
        to get an object that can be used as a Keras data generator.

        The generator should be given the ``(src,dst)`` node types using

        :param Graph: Graph Object suited for bipartite graph structure
        :param batch_size (int)
        :param num_samples (list): List of number of neighbor node sampler per GraphSAGE layer (hop) to take
        :param head_node_types (list): List of the types of the two head nodes forming the node pair
         For now, the only possible choice is as follows: ['user', 'item']
        :param seed (int or str, optional): Random seed for the sampling methods
        """
        super().__init__(Graph, batch_size)
        self.graph = Graph
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.head_node_types = head_node_types    # ['user', 'item']

        # Create sampling schema
        self._sampling_schema = [
            [('user', [0]), ('item', []), ('item', [1]), ('user', []),
             ('user', list(range(2, num_samples[0]+2, 1))), ('item', [])],
            [('user', []), ('item', [0]), ('item', []), ('user', [1]),
              ('user', []), ('item', list(range(2, num_samples[0]+2, 1)))],
        ]

        self.sampler = BreadthFirstWalker(Graph=Graph, num_of_walks=num_samples)

    def _get_features(self, node_samples, head_size):
        """
        Collect features from sampled nodes.
        Args:
            node_samples: A list of lists of node IDs
            head_size: The number of head nodes (typically the batch size).

        Returns:
            A list of numpy arrays that store the features for each head
            node.
        """
        # Note the if there are no samples for a node a zero array is returned.
        # Resize features to (batch_size, n_neighbours, feature_size)
        # for each node type (note that we can have different feature size for each node type)
        # G.node_features(['u_630'], 'user') -> (1, 24)
        # 한 번에 한 node_type 만 가능
        """
       batch_feats = [
            self.graph.node_features(layer_nodes, nt, use_ilocs=use_ilocs)
            for nt, layer_nodes in node_samples]
        """
        batch_feats = [
            self.graph.get_node_features_from_node(layer_nodes, nt)
            for nt, layer_nodes in node_samples
        ]

        # Resize features to (batch_size, n_neighbours, feature_size)
        batch_feats = [np.reshape(a, (head_size, -1, a.shape[1])) for a in batch_feats]

        return batch_feats

    def sample_features(self, head_links, batch_num):
        """
        Sample neighbours recursively from the head nodes, collect the features of the
        sampled nodes, and return these as a list of feature arrays for the GraphSAGE
        algorithm.

        Args:
            head_links (list): An iterable of edges to perform sampling for.
            batch_num (int): Batch number -> 이거 어디다 쓰냐?

        Returns:
            A list of the same length as `num_samples` of collected features from
            the sampled nodes of shape: ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where ``num_sampled_at_layer`` is the cumulative product of `num_samples`
            for that layer.
        """
        nodes_by_type = []
        # 2번 하는 이유
        # 한 번은 user, 한 번은 item
        for ii in range(2):
            # Extract head nodes from edges: each edge is a tuple of 2 nodes
            # so we are extracting 2 head nodes per edge
            head_nodes = [e[ii] for e in head_links]

            # Get sampled nodes for the sub-graphs starting from the (src, dst) head nodes
            # nodes_samples is list of two lists: [[samples for src], [samples for dst]]
            # node_samples = self.sampler.run(nodes=head_nodes, n=1, n_size=self.num_samples)
            node_samples = self.sampler.run_breadth_first_walk(nodes=head_nodes)

            # Reshape node samples to the required format for the HinSAGE model
            # This requires grouping the sampled nodes by edge type and in order
            nodes_by_type.append(
                [
                    (
                        nt,
                        reduce(
                            operator.concat,
                            (samples[ks] for samples in node_samples for ks in indices),
                            [],
                        ),
                    )
                    for nt, indices in self._sampling_schema[ii]
                ]
            )

        # Interlace the two lists
        # nodes_by_type[0] (for src head nodes) and nodes_by_type[1] (for dst head nodes)
        nodes_by_type = [
            tuple((ab[0][0], reduce(operator.concat, (ab[0][1], ab[1][1]))))
            for ab in zip(nodes_by_type[0], nodes_by_type[1])
        ]

        batch_feats = self._get_features(nodes_by_type, len(head_links))

        return batch_feats

    @property
    def type_adjacency_list(self):
        type_adjacency_list = [
            ('user', [2]), ('item', [3]), ('item', [4]), ('user', [5]),
            ('user', []), ('item', [])
        ]
        return type_adjacency_list



"""
nodes_by_type interlace 하기 전에
nodes_by_type = [
 [('user', ['user_0', 'user_0', 'user_0', 'user_1', 'user_1']),
  ('item', []),
  ('item',
   ['item_2',
    'item_2',
    'item_2',
    'item_1',
    'item_0',
    'item_1',
    'item_0',
    'item_1',
    'item_1',
    'item_0',
    'item_2',
    'item_0',
    'item_2',
    'item_2',
    'item_1',
    'item_1',
    'item_2',
    'item_1',
    'item_1',
    'item_2']),
  ('user', []),
  ('user',
   ['user_0',
    'user_1',
    'user_1',
    'user_0',
    'user_1',
    'user_0',
    'user_0',
    'user_0',
    'user_0',
    'user_0',
    'user_0',
    'user_0',
    'user_0',
    'user_0',
    'user_0',
    'user_1',
    'user_1',
    'user_1',
    'user_0',
    'user_0',
    'user_1',
    'user_0',
    'user_0',
    'user_0',
    'user_1',
    'user_0',
    'user_1',
    'user_1',
    'user_1',
    'user_1',
    'user_0',
    'user_1',
    'user_1',
    'user_1',
    'user_1',
    'user_1',
    'user_1',
    'user_1',
    'user_1',
    'user_0']),
  ('item', [])],
 [('user', []),
  ('item', ['item_0', 'item_1', 'item_2', 'item_1', 'item_2']),
  ('item', []),
  ('user',
   ['user_0',
    'user_0',
    'user_0',
    'user_0',
    'user_1',
    'user_0',
    'user_0',
    'user_0',
    'user_0',
    'user_0',
    'user_0',
    'user_1',
    'user_1',
    'user_0',
    'user_1',
    'user_1',
    'user_1',
    'user_0',
    'user_0',
    'user_1']),
  ('user', []),
  ('item',
   ['item_1',
    'item_0',
    'item_2',
    'item_0',
    'item_1',
    'item_1',
    'item_0',
    'item_0',
    'item_1',
    'item_2',
    'item_0',
    'item_2',
    'item_0',
    'item_2',
    'item_2',
    'item_1',
    'item_1',
    'item_2',
    'item_1',
    'item_0',
    'item_0',
    'item_0',
    'item_1',
    'item_2',
    'item_1',
    'item_2',
    'item_1',
    'item_1',
    'item_2',
    'item_1',
    'item_2',
    'item_2',
    'item_1',
    'item_2',
    'item_2',
    'item_1',
    'item_0',
    'item_1',
    'item_1',
    'item_1'])]]

"""


