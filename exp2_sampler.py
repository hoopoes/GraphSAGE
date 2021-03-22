# Experiments
import numpy as np
import pandas as pd
from typing import Dict, List
from collections import deque

class Sampler:
    def __init__(
        self,
        Graph,
        walk_length: int=None,
        num_of_walks: int=1,
        seed=None
    ):
        self.graph = Graph
        self.walk_length = walk_length
        self.num_of_walks = num_of_walks
        self.seed = seed

    def __repr__(self):
        return "Sampler Object"

    def run_breadth_first_walk(self, nodes: List=None):
        walk_length = self.walk_length
        num_of_walks = self.num_of_walks
        seed = self.seed

        walks = []
        depth = 2

        # node = 0 으로 놓고 해보자

        for node in nodes:      # iterate over root nodes
            for _ in range(num_of_walks):  # do n bounded breadth first walks from each root node
                queue = list()      # the queue of neighbours
                walk = list()       # the list of nodes in the subgraph of node

                # Start the walk by adding the head node, and node type to the frontier list q
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
                    if depth <= depth:
                        # Find edge types for current node type
                        et = self.graph_schema.schema[current_node_type]

                        neighbors = graph.get_neighbors_from_node(current_node)
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
                        queue.extend(
                            [
                                (sampled_node, et.n2, depth)
                                for sampled_node in samples
                            ]
                        )

                # finished i-th walk from node so add it to the list of walks as a list
                walks.append(walk)

        return walks



"""
nodes (iterable, optional) The root nodes from which individual walks start.
    If not provided, all nodes in the graph are used.
length (int): Length of the walks for the default UniformRandomWalk walker. Length >= 2
number_of_walks (int): Number of walks from each root node for the default UniformRandomWalk walker.
seed (int, optional): Random seed for the default UniformRandomWalk walker.
walker (RandomWalk, optional): A RandomWalk object to use instead of the default UniformRandomWalk walker.

graph_list = {1: set([3, 4]),
              2: set([3, 4, 5]),
              3: set([1, 5]),
              4: set([1]),
              5: set([2, 6]),
              6: set([3, 5])}
root_node = 1

def BFS_with_adj_list(graph, root):
    visited = []
    queue = deque([root])

    while queue:
        n = queue.popleft()
        if n not in visited:
            visited.append(n)
            queue += graph[n] - set(visited)
    return visited
  
print(BFS_with_adj_list(graph_list, root_node))

"""









