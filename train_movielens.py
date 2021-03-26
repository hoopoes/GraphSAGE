# Train Process

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras import Model, optimizers, losses, metrics

from bipartite_graph import Graph
from generator import PairSAGEGenerator
from models import PairSAGE
from stellargraph.layer import link_regression

import stellargraph as sg
from stellargraph import datasets
from sklearn.model_selection import train_test_split
from time import perf_counter

# Setting
batch_size = 100
# Use 70% of edges for training, the rest for testing:
train_size = 0.7
test_size = 0.3

# Movielens Dataset
dataset = datasets.MovieLens()
G, edges_with_ratings = dataset.load()

# 편집
edges_with_ratings = edges_with_ratings.rename(
    {'user_id': 'user', 'movie_id': 'item', 'rating': 'y'}, axis=1)
edges_with_ratings['user'] = edges_with_ratings['user'].apply(
    lambda x: '_'.join(['user', x.split('_')[1]]))
edges_with_ratings['item'] = edges_with_ratings['item'].apply(
    lambda x: '_'.join(['item', x.split('_')[1]]))

# --------
import os, sys
import itertools

# 1) Movie Lens data preprocessing
base_path = os.path.join(os.getcwd(), 'data')
user_path = os.path.join(base_path, 'users.dat')
item_path = os.path.join(base_path, 'movies.dat')
rating_path = os.path.join(base_path, 'ratings.dat')

def movie_preprocessing(movie):
    movie_col = list(movie.columns)
    movie_tag = [doc.split('|') for doc in movie['tag']]
    tag_table = {token: idx for idx, token in enumerate(set(itertools.chain.from_iterable(movie_tag)))}
    movie_tag = pd.DataFrame(movie_tag)
    tag_table = pd.DataFrame(tag_table.items())
    tag_table.columns = ['Tag', 'Index']

    # use one-hot encoding for movie genres (here called tag)
    tag_dummy = np.zeros([len(movie), len(tag_table)])

    for i in range(len(movie)):
        for j in range(len(tag_table)):
            if tag_table['Tag'][j] in list(movie_tag.iloc[i, :]):
                tag_dummy[i, j] = 1

    # combine the tag_dummy one-hot encoding table to original movie files
    movie = pd.concat([movie, pd.DataFrame(tag_dummy)], 1)
    movie_col.extend(['tag' + str(i) for i in range(len(tag_table))])
    movie.columns = movie_col
    movie = movie.drop('tag', 1)
    return movie

# user, item preprocess
items = pd.read_table(item_path, sep='::', names=['movie_id', 'movie_name', 'tag'], engine='python')
items = movie_preprocessing(items)
items = items.drop('movie_name', axis=1)
items = items.rename({'movie_id': 'item'}, axis=1)

users = pd.read_table(
    user_path, sep='::', names=['user_id', 'sex', 'age', 'occupation', 'post_code'],
    engine='python')
users['age_bin'] = pd.cut(
    users['age'], bins=[0, 20, 30, 40, 50, 60], labels=['10대', '20대', '30대', '40대', '50대'])
users = users.drop(['post_code', 'age'], axis=1)
users['occupation'] = users['occupation'].astype('str')
users = pd.get_dummies(users)
users = users.rename({'user_id': 'user'}, axis=1)


# edge
rating = pd.read_table(
    rating_path, sep="::", names=["user_id", "movie_id", "rating", "timestamp"], engine='python')
rating = rating.drop(['timestamp'], axis=1)
rating = rating.rename(
    {'user_id': 'user', 'movie_id': 'item', 'rating': 'y'}, axis=1)
print(rating.shape)

# 1 -> 0
def correct_id(df, col):
    df[col] = df[col] - 1
    df[col] = df[col].apply(lambda x: '_'.join([str(col), str(x)]))
    return df

users = correct_id(users, 'user')
items = correct_id(items, 'item')
rating = correct_id(rating, 'user')
rating = correct_id(rating, 'item')

# Graph Input
node_features = {
    'user': users.drop('user', axis=1).values,
    'item': items.drop('item', axis=1).values
}

edges = rating[['user', 'item']].values.T
flipped_edges = np.flip(edges, axis=0)
edges = np.concatenate([edges, flipped_edges], axis=1)

a = sorted(users['user'].unique())
b = sorted(items['item'].unique())

node_dict = {
    user_id: user_index for user_id, user_index
    in zip(a, list(range(0, len(a), 1)))}
add = {
    item_id: item_index for item_id, item_index
    in zip(b, list(range(len(a), len(a)+len(b), 1)))}
node_dict.update(add)


graph = Graph(node_features=node_features, edges=edges, node_dict=node_dict)

edges_train, edges_test = train_test_split(
    rating, train_size=train_size, test_size=test_size)

# user-item edge 리스트
# 아래가 link_ids
edgelist_train = list(edges_train[["user", "item"]].itertuples(index=False))
edgelist_test = list(edges_test[["user", "item"]].itertuples(index=False))
labels_train = edges_train["y"]
labels_test = edges_test["y"]


# ------
# Data Generator: batch_size 200으로 설정함
# generator: PairSAGE Generator
# train_gen, test_gen: LinkSequence
num_samples = [8, 4]
generator = PairSAGEGenerator(
    graph, batch_size, num_samples, head_node_types=["user", "item"])

# edge_list_train[0] = link_ids[0] = Pandas(user_id='u_630', movie_id='m_832')
train_gen = generator.flow(
    link_ids=edgelist_train, targets=labels_train, shuffle=True)
test_gen = generator.flow(
    link_ids=edgelist_test, targets=labels_test)

inputs = next(iter(train_gen))
print([inputs[0][i].shape for i in range(len(inputs[0]))])
# [(2, 1, 1),    (2, 1, 2),    (2, 4, 2),    (2, 4, 1),    (2, 8, 1),    (2, 8, 2)]
# [(200, 1, 24), (200, 1, 19), (200, 4, 19), (200, 4, 24), (200, 8, 24), (200, 8, 19)]

# models
pairsage = PairSAGE(layer_sizes=[16, 16], generator=generator)

x_inp, x_out = pairsage.in_out_tensors()
# x_out = [(None, 32), (None, 32)] - User, Item의 Embedding Matrix
score_prediction = link_regression(edge_embedding_method="concat")(x_out)

# Train
#def root_mean_square_error(s_true, s_pred):
#    return K.sqrt(K.mean(K.pow(s_true - s_pred, 2)))

model = Model(inputs=x_inp, outputs=score_prediction)
model.compile(
    optimizer=optimizers.Adam(lr=1e-2),
    loss=losses.mean_squared_error,
    metrics=[metrics.RootMeanSquaredError()])

num_workers = -1
epochs = 3

#test_metrics = model.evaluate(
#    test_gen, verbose=1, use_multiprocessing=False, workers=num_workers)

#print("Untrained model's Test Evaluation:")
#for name, val in zip(model.metrics_names, test_metrics):
#    print("\t{}: {:0.4f}".format(name, val))

history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=epochs,
    verbose=1,
    shuffle=False,
    use_multiprocessing=True,
    workers=num_workers)

# sg.utils.plot_history(history)

# ------
nodes = list(graph.node_dict.values())[0]
# graph.get_node_features_from_node(nodes, 'user')

t = generator.flow(edgelist_train[0:2], labels_train[0:2], shuffle=True)

inputs = next(iter(t))
print([inputs[0][i].shape for i in range(len(inputs[0]))])

history = model.fit(
    t, epochs=1, verbose=1, shuffle=False, use_multiprocessing=True, workers=num_workers)


start = perf_counter()
for i in range(90):
    out = next(iter(train_gen))
end = perf_counter() - start
print(end)









