# Generator
# stellagraph, pairsage generator

# 1) Generator
# 추상 클래스
import abc

class Generator(abc.ABC):
    # Creating Sequences for input for GNN
    @abc.abstractmethod
    def num_batch_dims(self):
        # Returns the number of batch dimensions in returned tensors (_not_ the batch size itself).
        # Ex) feature shape: (1, num_nodes, feature_size)
        pass

    @abc.abstractmethod
    def flow(self, *args, **kwargs):
        # Create a Tensorflow Keras Sequence for GNN
        pass

    def default_corrupt_input_index_groups(self):
        """
        Optionally returns the indices of input tensors that can be shuffled for
        :class:`.CorruptedGenerator` to use in :class:`.DeepGraphInfomax`.

        If this isn't overridden, this method returns None, indicating that the generator doesn't
        have a default or "canonical" set of indices that can be corrupted for Deep Graph Infomax.
        """
        return None


# 2) Batched Link Generator
# 필요 class
# StellarGraph, GraphSchema
# LinkSequence, OnDemandLinkSequence
# UnsupervisedSampler

from sequence import LinkSequence, OnDemandLinkSequence
from sampler import UnsupervisedSampler
from stellargraph.data import SampledHeterogeneousBreadthFirstWalk

# --------------
# (3) Batched Link Generator
import numpy as np
import operator
import collections
import abc
from functools import reduce

from utils import is_real_iterable, Aggregator
from stellargraph.core.graph import StellarGraph, GraphSchema


class BatchedLinkGenerator(Generator):
    def __init__(self, G, batch_size, schema=None, use_node_features=True):
        if not isinstance(G, StellarGraph):
            raise TypeError("Graph must be a StellarGraph or StellarDiGraph object.")

        self.graph = G
        self.batch_size = batch_size
        self.multiplicity = 2    # requires a model with 2 root nodes per query

        # We need a schema for compatibility with HinSAGE
        if schema is None:
            self.schema = G.create_graph_schema()
        elif isinstance(schema, GraphSchema):
            self.schema = schema
        else:
            raise TypeError("Schema must be a GraphSchema object")

        self.head_node_types = None
        self.sampler = None

        # Check if the graph has features
        if use_node_features:
            G.check_graph_for_ml()

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

        The targets are an array of numeric targets corresponding to the
        supplied node_ids to be used by the downstream task. They should
        be given in the same order as the list of node IDs.
        If they are not specified (for example, for use in prediction),
        the targets will not be available to the downstream task.

        Note that the shuffle argument should be True for training and
        False for prediction.

        Args:
            link_ids: an iterable of tuples of node IDs as (source, target)
            targets: a 2D array of numeric targets with shape
                ``(len(link_ids), target_size)``
            shuffle (bool): If True the links will be shuffled at each
                epoch, if False the links will be processed in order.
            seed (int, optional): Random seed

        Returns:
            A NodeSequence object to use with with StellarGraph models
            in Keras methods ``fit``, ``evaluate``,
            and ``predict``

        """
        if self.head_node_types is not None:
            expected_src_type = self.head_node_types[0]
            expected_dst_type = self.head_node_types[1]

        # Pass sampler to on-demand link sequence generation
        if isinstance(link_ids, UnsupervisedSampler):
            return OnDemandLinkSequence(self.sample_features, self.batch_size, link_ids)

        # Otherwise pass iterable (check?) to standard LinkSequence
        elif isinstance(link_ids, collections.abc.Iterable):
            # Check all IDs are actually in the graph and are of expected type
            for link in link_ids:
                if len(link) != 2:
                    raise KeyError("Expected link IDs to be a tuple of length 2")

                src, dst = link
                try:
                    node_type_src = self.graph.node_type(src)
                except KeyError:
                    raise KeyError(
                        f"Node ID {src} supplied to generator not found in graph"
                    )
                try:
                    node_type_dst = self.graph.node_type(dst)
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

            link_ids = [self.graph.node_ids_to_ilocs(ids) for ids in link_ids]

            return LinkSequence(
                self.sample_features,
                self.batch_size,
                link_ids,
                targets=targets,
                shuffle=shuffle,
                seed=seed,
            )

        else:
            raise TypeError(
                "Argument to .flow not recognised. "
                "Please pass a list of samples or a UnsupervisedSampler object."
            )

    def flow_from_dataframe(self, link_targets, shuffle=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            link_targets: a Pandas DataFrame of links specified by
                'source' and 'target' and an optional target label
                specified by 'label'.
            shuffle (bool): If True the links will be shuffled at each
                epoch, if False the links will be processed in order.

        Returns:
            A NodeSequence object to use with StellarGraph models
            in Keras methods ``fit``, ``evaluate``,
            and ``predict``

        """
        return self.flow(
            link_targets["source", "target"].values,
            link_targets["label"].values,
            shuffle=shuffle,
        )



class HinSAGELinkGenerator(BatchedLinkGenerator):
    """
    A data generator for link prediction with Heterogeneous HinSAGE models

    At minimum, supply the StellarGraph, the batch size, and the number of
    node samples for each layer of the GraphSAGE model.

    The supplied graph should be a StellarGraph object with node features for all node types.

    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    The generator should be given the ``(src,dst)`` node types using

    * It's possible to do link prediction on a graph where that link type is completely removed from the graph
      (e.g., "same_as" links in ER)

    .. seealso::

       Model using this generator: :class:`.HinSAGE`.

       Example using this generator: `link prediction <https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/hinsage-link-prediction.html>`__.

       Related functionality:

       - :class:`.UnsupervisedSampler` for unsupervised training using random walks
       - :class:`.HinSAGENodeGenerator` for node classification and related tasks
       - :class:`.GraphSAGELinkGenerator` for homogeneous graphs
       - :class:`.DirectedGraphSAGELinkGenerator` for directed homogeneous graphs

    Args:
        g (StellarGraph): A machine-learning ready graph.
        batch_size (int): Size of batch of links to return.
        num_samples (list): List of number of neighbour node samples per GraphSAGE layer (hop) to take.
        head_node_types (list, optional): List of the types (str) of the two head nodes forming the
            node pair. This does not need to be specified if ``G`` has only one node type.
        seed (int or str, optional): Random seed for the sampling methods.

    Example::

        G_generator = HinSAGELinkGenerator(G, 50, [10,10])
        data_gen = G_generator.flow(edge_ids)
    """

    def __init__(
        self,
        G,
        batch_size,
        num_samples,
        head_node_types=None,
        schema=None,
        seed=None,
        name=None,
    ):
        super().__init__(G, batch_size, schema)
        self.num_samples = num_samples
        self.name = name

        # This is a link generator and requires two nodes per query
        if head_node_types is None:
            # infer the head node types, if this is a homogeneous-node graph
            node_type = G.unique_node_type(
                "head_node_types: expected a pair of head node types because G has more than one node type, found node types: %(found)s"
            )
            head_node_types = [node_type, node_type]

        self.head_node_types = head_node_types
        if len(self.head_node_types) != 2:
            raise ValueError(
                "The head_node_types should be of length 2 for a link generator"
            )

        # Create sampling schema
        self._sampling_schema = self.schema.sampling_layout(
            self.head_node_types, self.num_samples
        )
        self._type_adjacency_list = self.schema.type_adjacency_list(
            self.head_node_types, len(self.num_samples)
        )

        # The sampler used to generate random samples of neighbours
        self.sampler = SampledHeterogeneousBreadthFirstWalk(
            G, graph_schema=self.schema, seed=seed
        )

    def _get_features(self, node_samples, head_size, use_ilocs=False):
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
        batch_feats = [
            self.graph.node_features(layer_nodes, nt, use_ilocs=use_ilocs)
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
            batch_num (int): Batch number

        Returns:
            A list of the same length as `num_samples` of collected features from
            the sampled nodes of shape: ``(len(head_nodes), num_sampled_at_layer, feature_size)``
            where ``num_sampled_at_layer`` is the cumulative product of `num_samples`
            for that layer.
        """
        nodes_by_type = []
        for ii in range(2):
            # Extract head nodes from edges: each edge is a tuple of 2 nodes, so we are extracting 2 head nodes per edge
            head_nodes = [e[ii] for e in head_links]

            # Get sampled nodes for the subgraphs starting from the (src, dst) head nodes
            # nodes_samples is list of two lists: [[samples for src], [samples for dst]]
            node_samples = self.sampler.run(
                nodes=head_nodes, n=1, n_size=self.num_samples
            )

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

        # Interlace the two lists, nodes_by_type[0] (for src head nodes) and nodes_by_type[1] (for dst head nodes)
        nodes_by_type = [
            tuple((ab[0][0], reduce(operator.concat, (ab[0][1], ab[1][1]))))
            for ab in zip(nodes_by_type[0], nodes_by_type[1])
        ]

        batch_feats = self._get_features(nodes_by_type, len(head_links), use_ilocs=True)

        return batch_feats



