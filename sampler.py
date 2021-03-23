# BreadFirstWalker Class

import numpy as np
from typing import List
from collections import deque

# TODO: list append, pop 대신 deque 사용
# TODO: seed 명시

class BreadthFirstWalker:
    def __init__(self, Graph, num_of_walks=None, seed=None):
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
        self.seed = seed
        self.walk_length = len(num_of_walks)

    def __repr__(self):
        identity = "<Breadth First Walker with {} number of walks>".format(self.num_of_walks)
        return identity

    def run_breadth_first_walk(self, nodes: List=None):
        walk_length = self.walk_length
        num_of_walks = self.num_of_walks

        walks = []
        # ex) nodes = ['user_0', 'user_31']

        for node in nodes:    # iterate over root nodes
            queue = []        # the queue of neighbours
            walk = []         # the list of nodes in the sub-graph of node

            # Start the walk by adding the head node, and node type to the frontier list queue
            node_type = self.graph.get_node_type(node)[0]
            queue.extend([(node, node_type, 0)])

            # add the root node to the walks
            walk.append([node])
            while len(queue) > 0:
                # remove the top element in the queue and pop the item from the front of the list
                frontier = queue.pop(0)
                current_node, current_node_type, depth = frontier
                depth = depth + 1  # the depth of the neighboring nodes

                # consider the sub-graph up to and including depth d from root node
                if depth <= walk_length:
                    neighbors = self.graph.get_neighbors_from_node(current_node)

                    # Neighbors might exist or not
                    if len(neighbors) > 0:
                        samples = np.random.choice(
                            neighbors, size=num_of_walks[depth-1], replace=True).tolist()
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


