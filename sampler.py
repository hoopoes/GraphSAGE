# BreadFirstWalker Class
from collections import namedtuple, deque
import random as rn
import numpy.random as np_rn
from typing import List


RandomState = namedtuple("RandomState", "random, numpy")

def _global_state():
    return RandomState(rn, np_rn)

def _seeded_state(s):
    return RandomState(rn.Random(s), np_rn.RandomState(s))

_rs = _global_state()

def random_state(seed):
    if seed is None:
        return _rs
    else:
        return _seeded_state(seed)

rs, _ = random_state(seed=None)

class BreadthFirstWalker:
    def __init__(self, Graph, num_of_walks=None):
        """
        Define BreadthFirstWalker
        This Object will be used in generating Generator

        :param Graph: Graph Class
        :param num_of_walks: number of walks per layer
         ex) [8, 4]
        """
        if not isinstance(num_of_walks, list):
            raise TypeError("num_of_walks must be list of integers")
        self.graph = Graph
        self.num_of_walks = num_of_walks
        self.rs = rs
        self.walk_length = len(num_of_walks)

    def __repr__(self):
        identity = "<Breadth First Walker with {} number of walks>".format(self.num_of_walks)
        return identity

    def run_breadth_first_walk(self, nodes: List=None):
        walk_length = self.walk_length
        num_of_walks = self.num_of_walks

        walks = []
        # ex) nodes = ['user_0', 'user_31']

        for node in nodes:      # iterate over root nodes
            walk = []           # the list of nodes in the sub-graph of node
            queue = deque()     # the queue of neighbours
            # queue = []

            # Start the walk by adding the head node, and node type to the frontier list queue
            node_type = self.graph.get_node_type(node)
            queue.extend([(node, node_type, 0)])

            # add the root node to the walks
            walk.append([node])
            while len(queue) > 0:
                # remove the top element in the queue and pop the item from the front of the list
                frontier = queue.popleft()
                # frontier = queue.pop(0)
                current_node, current_node_type, depth = frontier
                depth = depth + 1  # the depth of the neighboring nodes

                # consider the sub-graph up to and including depth d from root node
                if depth <= walk_length:
                    neighbors = self.graph.get_neighbors_from_node(current_node)

                    # Neighbors might exist or not
                    if len(neighbors) > 0:
                        samples = self.rs.choices(neighbors, k=num_of_walks[depth - 1])
                        #samples = np.random.choice(
                        #    neighbors, size=num_of_walks[depth-1], replace=True).tolist()
                    else:
                        raise ValueError("Samples must exist. Self-node also can be a sample")

                    walk.append(samples)
                    queue.extend(
                        [
                            (sampled_node, self.graph.get_node_type(sampled_node), depth)
                            for sampled_node in samples
                        ]
                    )

            # finished i-th walk from node so add it to the list of walks as a list
            # one walk - one node
            walks.append(walk)

        return walks
