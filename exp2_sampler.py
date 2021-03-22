# Experiments
import numpy as np
import pandas as pd
from typing import Dict, List
from collections import deque

walk_length = 2
num_of_walks = [4, 2]

class Sampler:
    def __init__(
        self,
        Graph,
        num_of_walks=None,    # list
        seed=None
    ):
        if not isinstance(num_of_walks, list):
            raise TypeError("num_of_walks must be list of integers")
        self.graph = Graph
        self.num_of_walks = num_of_walks
        self.seed = seed
        self.walk_length = len(num_of_walks)

    def __repr__(self):
        return "Sampler Object"

    def run_breadth_first_walk(self, nodes: List=None):
        walk_length = self.walk_length
        num_of_walks = self.num_of_walks

        walks = []
        # ex) nodes = [0, 3, 13]

        for node in nodes:    # iterate over root nodes
            queue = []        # the queue of neighbours
            walk = []         # the list of nodes in the subgraph of node

            # Start the walk by adding the head node, and node type to the frontier list queue
            node_type = self.graph.get_node_type(node)[0]
            queue.extend([(node, node_type, 0)])

            # add the root node to the walks
            walk.append([node])
            while len(queue) > 0:
                # remove the top element in the queue and pop the item from the front of the list
                frontier = queue.pop(0)
                current_node, current_node_type, depth = frontier
                depth = depth + 1  # the depth of the neighbouring nodes

                # consider the subgraph up to and including depth d from root node
                if depth <= walk_length:
                    neighbors = self.graph.get_neighbors_from_node(current_node)

                    # 이웃이 존재할 수도 있고 아닐 수도 있다.
                    if len(neighbors) > 0:
                        samples = np.random.choice(
                            neighbors, size=num_of_walks[depth-1], replace=True).tolist()
                    else:
                        raise ValueError("sample은 무조건 존재한다. 자기자신도 있으니까")

                    walk.append(samples)
                    queue.extend(
                        [
                            (sampled_node, self.graph.get_node_type(sampled_node), depth)
                            for sampled_node in samples
                        ]
                    )

            # finished i-th walk from node so add it to the list of walks as a list
            # walk 하나에 node 하나
            walks.append(walk)

        return walks


sampler = Sampler(Graph=graph, num_of_walks=num_of_walks)
nodes = [0, 1, 1, 0]
walks = sampler.run_breadth_first_walk(nodes=nodes)



