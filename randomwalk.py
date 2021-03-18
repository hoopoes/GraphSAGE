# Components of Generator
# 1) RandomWalk, UniformRandomWalk
#    GraphWalk, SampledHeterogeneousBreadthFirstWalk
# 2) UnsupervisedSampler
# 3) LinkSequence, OnDemandLinkSequence

# Unsupervised Sampler 정의를 위해 필요함

import numpy as np
import warnings
from abc import ABC, abstractmethod
from utils import random_state, is_real_iterable

from stellargraph.core.graph import StellarGraph, GraphSchema
# from stellargraph.data.explorer import GraphWalk, SampledHeterogeneousBreadthFirstWalk


__all__ = [
    "RandomWalk",
    "UniformRandomWalk",
    "GraphWalk",
    "SampledHeterogeneousBreadthFirstWalk"
]


def _default_if_none(value, default, name, ensure_not_none=True):
    value = value if value is not None else default
    if ensure_not_none and value is None:
        raise ValueError(
            f"{name}: expected a value to be specified in either `__init__` or `run`, found None in both"
        )
    return value


class RandomWalk(ABC):
    """
    Abstract base class for Random Walk classes.
    A Random Walk class must implement a `run` method
    which takes an iterable of node IDs and returns a list of walks.
    Each walk is a list of node IDs that contains the starting node as its first element.
    """

    def __init__(self, graph, seed=None):
        if not isinstance(graph, StellarGraph):
            raise TypeError("Graph must be a StellarGraph or StellarDiGraph.")

        self.graph = graph
        self._random_state, self._np_random_state = random_state(seed)

    def _get_random_state(self, seed):
        """
        Args:
            seed: The optional seed value for a given run.

        Returns:
            The random state as determined by the seed.
        """
        if seed is None:
            # Restore the random state
            return self._random_state, self._np_random_state
        return random_state(seed)

    @staticmethod
    def _validate_walk_params(nodes, n: int, length: int):
        if not is_real_iterable(nodes):
            raise ValueError(f"nodes: expected an iterable, found: {nodes}")
        if len(nodes) == 0:
            warnings.warn(
                "No root node IDs given. An empty list will be returned as a result.",
                RuntimeWarning,
                stacklevel=3,
            )

    @abstractmethod
    def run(self, nodes, **kwargs):
        pass


class UniformRandomWalk(RandomWalk):
    """
    graph (StellarGraph): Graph to traverse
    n (int, optional): Total number of random walks per root node
    length (int, optional): Maximum length of each random walk
    seed (int, optional): Random number generator seed
    """

    def __init__(self, graph, n=None, length=None, seed=None):
        super().__init__(graph, seed=seed)
        self.n = n
        self.length = length

    def run(self, nodes, n=None, length=None, seed=None):
        """
        Perform a random walk starting from the root nodes. Optional parameters default to using the
        values passed in during construction.

        Args:
            nodes (list): The root nodes as a list of node IDs
            n (int, optional): Total number of random walks per root node
            length (int, optional): Maximum length of each random walk
            seed (int, optional): Random number generator seed

        Returns:
            List of lists of nodes ids for each of the random walks

        """
        n = _default_if_none(n, self.n, "n")
        length = _default_if_none(length, self.length, "length")
        self._validate_walk_params(nodes, n, length)
        rs, _ = self._get_random_state(seed)

        nodes = self.graph.node_ids_to_ilocs(nodes)

        # for each root node, do n walks
        return [self._walk(rs, node, length) for node in nodes for _ in range(n)]

    def _walk(self, rs, start_node, length):
        walk = [start_node]
        current_node = start_node
        for _ in range(length - 1):
            neighbours = self.graph.neighbor_arrays(current_node, use_ilocs=True)
            if len(neighbours) == 0:
                # dead end, so stop
                break
            else:
                # has neighbours, so pick one to walk to
                current_node = rs.choice(neighbours)
            walk.append(current_node)

        return list(self.graph.node_ilocs_to_ids(walk))


"""
Example

from stellargraph import datasets
dataset = datasets.MovieLens()
G, edges_with_ratings = dataset.load()
nodes = list(G.nodes())
graph_schema = G.create_graph_schema(nodes)

# n: number of walks -> 각 node 별로 몇 번 walk를 만들 것인가
# length: walk의 길이
# 조건: n >= 1, lengh >= 2
walker = UniformRandomWalk(G, n=32, length=8, seed=0)
walks = walker.run(nodes, n=3, length=3, seed=0)

[['m_1', 'u_597', 'm_151'],
 ['m_1', 'u_745', 'm_10'],
 ['m_1', 'u_514', 'm_177'],
 ['m_2', 'u_807', 'm_498'],
 ['m_2', 'u_110', 'm_212'],
 ['m_2', 'u_435', 'm_818'],
 ['m_3', 'u_450', 'm_1269'],
 ['m_3', 'u_459', 'm_405'],
 ['m_3', 'u_859', 'm_287']]
"""


def naive_weighted_choices(rs, weights, size=None):
    """
    Select indices at random, weighted by the iterator `weights` of
    arbitrary (non-negative) floats. That is, `x` will be returned
    with probability `weights[x]/sum(weights)`.

    For doing a single sample with arbitrary weights, this is much (5x
    or more) faster than numpy.random.choice, because the latter
    requires a lot of preprocessing (normalized probabilities), and
    does a lot of conversions/checks/preprocessing internally.
    """
    probs = np.cumsum(weights)
    total = probs[-1]
    if total == 0:
        # all weights were zero (probably), so we shouldn't choose anything
        return None

    thresholds = rs.random() if size is None else rs.random(size)
    idx = np.searchsorted(probs, thresholds * total, side="left")

    return idx

"""
GraphWalk에서
adj = G._adjacency_types(graph_schema, use_ilocs=True)
a = adj[list(adj.keys())[0]] # 943, Item
b = adj[list(adj.keys())[1]] # 1682, User

b[0] = [1682, 1683, 1686, ..., 2622] -> Link Information

"""

class GraphWalk(object):
    # Base class for exploring graphs

    def __init__(self, graph, graph_schema=None, seed=None):
        self.graph = graph

        # Initialize the random state
        self._check_seed(seed)
        self._random_state, self._np_random_state = random_state(seed)

        # We require a StellarGraph for this
        if not isinstance(graph, StellarGraph):
            raise TypeError("Graph must be a StellarGraph or StellarDiGraph.")

        if not graph_schema:
            self.graph_schema = self.graph.create_graph_schema()
        else:
            self.graph_schema = graph_schema

        if type(self.graph_schema) is not GraphSchema:
            self._raise_error(
                "The parameter graph_schema should be either None or of type GraphSchema.")

    def get_adjacency_types(self):
        # Allow additional info for heterogeneous graphs.
        adj = getattr(self, "adj_types", None)
        if not adj:
            # Create a dict of adjacency lists per edge type, for faster neighbour sampling from graph in SampledHeteroBFS:
            self.adj_types = adj = self.graph._adjacency_types(self.graph_schema, use_ilocs=True)
        return adj

    def _check_seed(self, seed):
        if seed is not None:
            if type(seed) != int:
                self._raise_error(
                    "The random number generator seed value, seed, should be integer type or None.")
            if seed < 0:
                self._raise_error(
                    "The random number generator seed value, seed, should be non-negative integer or None.")

    def _get_random_state(self, seed):
        """
        Args:
            seed: The optional seed value for a given run.

        Returns:
            The random state as determined by the seed.
        """
        if seed is None:
            # Use the class's random state
            return self._random_state, self._np_random_state
        # seed the random number generators
        return random_state(seed)

    def neighbors(self, node):
        return self.graph.neighbor_arrays(node, use_ilocs=True)

    def run(self, *args, **kwargs):
        """
        To be overridden by subclasses..
        It should return the sequences of nodes in each random walk.
        """
        raise NotImplementedError

    def _raise_error(self, msg):
        raise ValueError("({}) {}".format(type(self).__name__, msg))

    def _check_common_parameters(self, nodes, n, length, seed):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            nodes: <list> A list of root node ids from which to commence the random walks.
            n: <int> Number of walks per node id.
            length: <int> Maximum length of each walk.
            seed: <int> Random number generator seed.
        """
        self._check_nodes(nodes)
        self._check_repetitions(n)
        self._check_length(length)
        self._check_seed(seed)

    def _check_nodes(self, nodes):
        if nodes is None:
            self._raise_error("A list of root node IDs was not provided.")
        if not is_real_iterable(nodes):
            self._raise_error("Nodes parameter should be an iterable of node IDs.")
        if (
            len(nodes) == 0
        ):  # this is not an error but maybe a warning should be printed to inform the caller
            warnings.warn(
                "No root node IDs given. An empty list will be returned as a result.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _check_repetitions(self, n):
        if type(n) != int:
            self._raise_error(
                "The number of walks per root node, n, should be integer type."
            )
        if n <= 0:
            self._raise_error(
                "The number of walks per root node, n, should be a positive integer."
            )

    def _check_length(self, length):
        if type(length) != int:
            self._raise_error("The walk length, length, should be integer type.")
        if length <= 0:
            # Technically, length 0 should be okay, but by consensus is invalid.
            self._raise_error("The walk length, length, should be a positive integer.")

    # For neighbourhood sampling
    def _check_sizes(self, n_size):
        err_msg = "The neighbourhood size must be a list of non-negative integers."
        if not isinstance(n_size, list):
            self._raise_error(err_msg)
        if len(n_size) == 0:
            # Technically, length 0 should be okay, but by consensus it is invalid.
            self._raise_error("The neighbourhood size list should not be empty.")
        for d in n_size:
            if type(d) != int or d < 0:
                self._raise_error(err_msg)

    def _sample_neighbours_untyped(
        self, neigh_func, py_and_np_rs, cur_node, size, weighted
    ):
        """
        Sample ``size`` neighbours of ``cur_node`` without checking node types or edge types,
        optionally using edge weights.
        """
        if cur_node != -1:
            neighbours = neigh_func(
                cur_node, use_ilocs=True, include_edge_weight=weighted)

            if weighted:
                neighbours, weights = neighbours
        else:
            neighbours = []

        if len(neighbours) > 0:
            if weighted:
                # sample following the edge weights
                idx = naive_weighted_choices(py_and_np_rs[1], weights, size=size)
                if idx is not None:
                    return neighbours[idx]
            else:
                # uniform sample; for small-to-moderate `size`s (< 100 is typical for GraphSAGE), random
                # has less overhead than np.random
                return np.array(py_and_np_rs[0].choices(neighbours, k=size))

        # no neighbours (e.g. isolated node, cur_node == -1 or all weights 0), so propagate the -1 sentinel
        return np.full(size, -1)


class SampledHeterogeneousBreadthFirstWalk(GraphWalk):
    """
    Breadth First Walk for heterogeneous graphs that generates a sampled number of paths from a starting node.
    It can be used to extract a random sub-graph starting from a set of initial nodes.
    """

    def run(self, nodes, n_size, n=1, seed=None):
        """
        Performs a sampled breadth-first walk starting from the root nodes.

        Args:
            nodes (list): A list of root node ids such that from each node n BFWs will be generated
                with the number of samples per hop specified in n_size.
            n_size (int): The number of neighbouring nodes to expand at each depth of the walk. Sampling of
            n (int, default 1): Number of walks per node id. Neighbours with replacement is always used regardless
                of the node degree and number of neighbours requested.
            seed (int, optional): Random number generator seed; default is None

        Returns:
            A list of lists such that each list element is a sequence of ids corresponding to a sampled Heterogeneous
            BFW.
        """
        self._check_sizes(n_size)
        self._check_common_parameters(nodes, n, len(n_size), seed)
        rs, _ = self._get_random_state(seed)

        adj = self.get_adjacency_types()

        walks = []
        d = len(n_size)  # depth of search

        for node in nodes:      # iterate over root nodes
            for _ in range(n):  # do n bounded breadth first walks from each root node
                q = list()      # the queue of neighbours
                walk = list()   # the list of nodes in the subgraph of node

                # Start the walk by adding the head node, and node type to the frontier list q
                node_type = self.graph.node_type(node, use_ilocs=True)
                q.extend([(node, node_type, 0)])

                # add the root node to the walks
                walk.append([node])
                while len(q) > 0:
                    # remove the top element in the queue and pop the item from the front of the list
                    frontier = q.pop(0)
                    current_node, current_node_type, depth = frontier
                    depth = depth + 1  # the depth of the neighbouring nodes

                    # consider the subgraph up to and including depth d from root node
                    if depth <= d:
                        # Find edge types for current node type
                        current_edge_types = self.graph_schema.schema[current_node_type]

                        # Create samples of neighbours for all edge types
                        for et in current_edge_types:
                            neigh_et = adj[et][current_node]

                            # If there are no neighbours of this type then we return None
                            # in the place of the nodes that would have been sampled
                            # YT update: with the new way to get neigh_et from adj[et][current_node], len(neigh_et) is always > 0.
                            # In case of no neighbours of the current node for et, neigh_et == [None],
                            # and samples automatically becomes [None]*n_size[depth-1]
                            if len(neigh_et) > 0:
                                samples = rs.choices(neigh_et, k=n_size[depth - 1])
                            else:  # this doesn't happen anymore, see the comment above
                                _size = n_size[depth - 1]
                                samples = [-1] * _size

                            walk.append(samples)
                            q.extend(
                                [
                                    (sampled_node, et.n2, depth)
                                    for sampled_node in samples
                                ]
                            )

                # finished i-th walk from node so add it to the list of walks as a list
                walks.append(walk)

        return walks


