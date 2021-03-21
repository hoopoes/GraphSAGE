# Experiments
import numpy as np
import pandas as pd
from typing import Dict, List

class RandomWalker():





class Sampler:
    def __init__(
        self,
        Graph,
        nodes: List=None,
        walk_length: int=None,
        num_of_walks: int=None,
        seed=None,
        walker=None
    ):
        self.graph = Graph
        self.nodes = nodes
        self.walk_length = walk_length
        self.num_of_walks = num_of_walks
        self.seed = seed
        self.walker = walker

    def __repr__(self):
        return "Sampler Object"



"""
nodes (iterable, optional) The root nodes from which individual walks start.
    If not provided, all nodes in the graph are used.
length (int): Length of the walks for the default UniformRandomWalk walker. Length >= 2
number_of_walks (int): Number of walks from each root node for the default UniformRandomWalk walker.
seed (int, optional): Random seed for the default UniformRandomWalk walker.
walker (RandomWalk, optional): A RandomWalk object to use instead of the default UniformRandomWalk walker.
"""









