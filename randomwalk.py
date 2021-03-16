# RandomWalk
# UniformRandomWalk
# Unsupervised Sampler 정의를 위해 필요함

import numpy as np
import warnings
from abc import ABC, abstractmethod
from utils import random_state, is_real_iterable
# from stellargraph.random import random_state

from stellargraph.core.graph import StellarGraph


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
