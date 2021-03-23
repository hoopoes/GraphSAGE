# Train Process

import numpy as np
import pandas as pd

from bipartite_graph import Graph
from sampler import BreadthFirstWalker

users = np.array([[-1], [4]])
items = np.array([[0.4, 100], [0.1, 200], [0.9, 300]])
node_features = {'user': users, 'item': items}

#edges = np.array([[0, 0, 0, 1, 1, 2, 3, 3, 4, 4],
#                  [2, 3, 4, 3, 4, 0, 0, 1, 0, 1]])
edges = np.array([['user_0', 'user_0', 'user_0', 'user_1', 'user_1',
                   'item_0', 'item_1', 'item_2', 'item_1', 'item_2'],
                  ['item_0', 'item_1', 'item_2', 'item_1', 'item_2',
                   'user_0', 'user_0', 'user_0', 'user_1', 'user_1']])

node_dict = {'user_0': 0, 'user_1': 1, 'item_0': 2, 'item_1': 3, 'item_2': 4}

df = pd.DataFrame(edges.T).rename({0: 'user', 1: 'item'}, axis=1).head(5)
df['y'] = [1, 0, 0, 1, 0]
train_edge = list(df[["user", "item"]].itertuples(index=False))
train_label  = list(df['y'])

graph = Graph(node_features=node_features, edges=edges, node_dict=node_dict)

walk_length = 2
num_of_walks = [4, 2]

sampler = BreadthFirstWalker(Graph=graph, num_of_walks=num_of_walks)
# nodes = ['user_0', 'user_1', 'user_1', 'user_0']
nodes = [0, 1, 1, 0]
walks = sampler.run_breadth_first_walk(nodes=nodes)









