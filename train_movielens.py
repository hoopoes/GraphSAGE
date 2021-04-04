# Train Process
import os, sys
import itertools
import numpy as np
import pandas as pd
from tensorflow.keras import Model, optimizers, losses, metrics

from bipartite_graph import Graph
from generator import PairSAGEGenerator
from models import PairSAGE
from link_utils import link_classification

from sklearn.model_selection import train_test_split

# Setting
batch_size = 100
# Use 70% of edges for training, the rest for testing:
train_size = 0.7
test_size = 0.3

# Movielens Dataset
# dataset = datasets.MovieLens()
# G, edges_with_ratings = dataset.load()


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

# Rating -> 0, 1
def change_to_binary(x):
    if x <= 2:
        return 0
    elif x >= 5:
        return 1
    else:
        return -1

rating['y'] = rating['y'].apply(lambda x: change_to_binary(x))
rating = rating[rating['y'].isin([0, 1])].drop_duplicates()

# 6014, 3232
valid_users = list(rating[rating['y'] == 1]['user'].unique())
valid_items = list(rating[rating['y'] == 1]['item'].unique())

users = users[users['user'].isin(valid_users)]
items = items[items['item'].isin(valid_items)]
rating = rating[(rating['user'].isin(valid_users)) & (rating['item'].isin(valid_items))]

# id 전처리
users['user'] = users['user'].apply(lambda x: '_'.join(['u', str(x)]))
user_ids = sorted(list(users['user'].unique()))
new_user_ids = {user_ids[i]: i for i in range(len(user_ids))}
users['user'] = users['user'].map(new_user_ids)

items['item'] = items['item'].apply(lambda x: '_'.join(['i', str(x)]))
item_ids = sorted(list(items['item'].unique()))
new_item_ids = {item_ids[i]: i for i in range(len(item_ids))}
items['item'] = items['item'].map(new_item_ids)

rating['user'] = rating['user'].apply(lambda x: '_'.join(['u', str(x)]))
rating['user'] = rating['user'].map(new_user_ids)
rating['item'] = rating['item'].apply(lambda x: '_'.join(['i', str(x)]))
rating['item'] = rating['item'].map(new_item_ids)


max_user_index = rating['user'].max()
items['item'] = items['item'] + max_user_index + 1
rating['item'] = rating['item'] + max_user_index + 1


# Graph Input
# TODO Padding?
node_features = {
    'user': users.drop('user', axis=1).values,
    'item': items.drop('item', axis=1).values
}

# All users, items
a = sorted(users['user'].unique())
b = sorted(items['item'].unique())

# Edges: only count 1 value
edges = rating[rating['y'] == 1][['user', 'item']].drop_duplicates().values.T
flipped_edges = np.flip(edges, axis=0)
edges = np.concatenate([edges, flipped_edges], axis=1)

# Nodes
nodes = {'user': a, 'item': b}


# Graph Class 생성
graph = Graph(node_features=node_features, nodes=nodes, edges=edges)

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
pairsage = PairSAGE(layer_sizes=[32, 32], generator=generator)

x_inp, x_out = pairsage.in_out_tensors()
# x_out = [(None, 32), (None, 32)] - User, Item의 Embedding Matrix
score_prediction = link_classification(edge_embedding_method="concat")(x_out)

# Train

model = Model(inputs=x_inp, outputs=score_prediction)
model.compile(
    optimizer=optimizers.Adam(lr=1e-2),
    loss=losses.mean_squared_error,
    metrics=[metrics.RootMeanSquaredError(),
             metrics.BinaryAccuracy(),
             metrics.AUC()])

num_workers = -1
epochs = 1

# Untrained Model Performance
#test_metrics = model.evaluate(
#    test_gen, verbose=1, use_multiprocessing=False, workers=num_workers)

#print("Untrained model's Test Evaluation:")
#for name, val in zip(model.metrics_names, test_metrics):
#    print("\t{}: {:0.4f}".format(name, val))

# Train
print("Positive Value Ratio: {:.4f}".format(
    list(rating.groupby('y')['y'].agg('size'))[1] / np.sum(list(
        rating.groupby('y')['y'].agg('size')))))

history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=epochs,
    verbose=1,
    shuffle=False,
    use_multiprocessing=True,
    workers=num_workers)



