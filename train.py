# Train Process

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras import Model, optimizers, losses, metrics

from bipartite_graph import Graph
from sampler import BreadthFirstWalker
from generator import PairSAGEGenerator
from models import PairSAGE
from stellargraph.layer import link_regression

import stellargraph as sg


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

# generator
num_samples = [4, 2]
gen = PairSAGEGenerator(graph, 2, num_samples, ['user', 'item'])
train = gen.flow(train_edge, train_label, shuffle=True)
print(train.batch_size, train.data_size, train.ids, train.indices, train.targets)

inputs = next(iter(train))

print([inputs[0][i].shape for i in range(len(inputs[0]))])
# [(2, 1, 1),    (2, 1, 2),    (2, 4, 2),    (2, 4, 1),    (2, 8, 1),    (2, 8, 2)]
# [(200, 1, 24), (200, 1, 19), (200, 4, 19), (200, 4, 24), (200, 8, 24), (200, 8, 19)]

# models
pairsage = PairSAGE(layer_sizes=[32, 32], generator=gen)

x_inp, x_out = pairsage.in_out_tensors()
# x_out = [(None, 32), (None, 32)] - User, Item의 Embedding Matrix
score_prediction = link_regression(edge_embedding_method="concat")(x_out)

# Train
def root_mean_square_error(s_true, s_pred):
    return K.sqrt(K.mean(K.pow(s_true - s_pred, 2)))

model = Model(inputs=x_inp, outputs=score_prediction)
model.compile(
    optimizer=optimizers.Adam(lr=1e-2),
    loss=losses.mean_squared_error,
    metrics=[root_mean_square_error, metrics.mae],
)

num_workers = 4
epochs = 5

#test_metrics = model.evaluate(
#    test_gen, verbose=1, use_multiprocessing=False, workers=num_workers)

#print("Untrained model's Test Evaluation:")
#for name, val in zip(model.metrics_names, test_metrics):
#    print("\t{}: {:0.4f}".format(name, val))


history = model.fit(
    train,
    epochs=epochs,
    verbose=1,
    shuffle=False,
    use_multiprocessing=False,
    workers=num_workers)

sg.utils.plot_history(history)





