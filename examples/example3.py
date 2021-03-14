# HinSAGE Example
import json
import pandas as pd
import numpy as np
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error

import stellargraph as sg
from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import HinSAGE, link_regression
from tensorflow.keras import Model, optimizers, losses, metrics
import tensorflow.keras.backend as K

import multiprocessing
from stellargraph import datasets
import matplotlib.pyplot as plt


batch_size = 200
epochs = 20
# Use 70% of edges for training, the rest for testing:
train_size = 0.7
test_size = 0.3

dataset = datasets.MovieLens()
G, edges_with_ratings = dataset.load()
# G.info()

edges_train, edges_test = model_selection.train_test_split(
    edges_with_ratings, train_size=train_size, test_size=test_size
)

"""
edges_train example:

user_id movie_id  rating
u_500    m_660       2
u_847    m_141       3
"""

# user-item edge 리스트
# 70000, 30000
edgelist_train = list(edges_train[["user_id", "movie_id"]].itertuples(index=False))
edgelist_test = list(edges_test[["user_id", "movie_id"]].itertuples(index=False))

# pd.Series
labels_train = edges_train["rating"]
labels_test = edges_test["rating"]

# length of num_samples list defines the number of layers/iterations in the HinSAGE
# 1차 이웃: 8개, 2차 이웃: 1차 이웃 당 4개씩
num_samples = [8, 4]

# Data Generator: batch_size 200으로 설정함
generator = HinSAGELinkGenerator(
    G, batch_size, num_samples, head_node_types=["user", "movie"]
)
train_gen = generator.flow(edgelist_train, labels_train, shuffle=True)
test_gen = generator.flow(edgelist_test, labels_test)

# Input 예시
out = next(iter(train_gen))
x_inp = out[0]
x_inp_shapes = [out[0][i].shape for i in range(6)]
print(x_inp_shapes)

# ex: [(200, 1, 24), (200, 1, 19), (200, 8, 19), (200, 8, 24), (200, 32, 24), (200, 32, 19)]
# 맨앞 2개: 최초의 Input Node를 의미하며 User 200개, Item 200개를 추출함
#  - 이는 User-Item Pair 200개를 의미하며 labels_train을 보면 200개의 True Rating 값을 얻을 수 있음
# 중간 2개: 위 Node의 1차 이웃, 각 Input Node에 대하여 8개씩 추출함
#  - 순서가 맨 앞은 24, 19 -> User, Item
#  - 중간의 경우 19, 24 -> Item, User를 의미함
# 맨뒤 2개: 위 1차 이웃에 대하여 다시 4개씩 2차 이웃을 뽑음
#  - 24, 19이므로 다시 User, Item 순서를 의미함


# Model
hinsage_layer_sizes = [32, 32]

# 이제 아래 HinSAGE 모델 구조를 정확히 알아야 한다.
hinsage = HinSAGE(
    layer_sizes=hinsage_layer_sizes, generator=generator, bias=True, dropout=0.0
)


x_inp, x_out = hinsage.in_out_tensors()
# x_out = [(None, 32), (None, 32)] - User, Item의 Embedding Matrix
score_prediction = link_regression(edge_embedding_method="concat")(x_out)

def root_mean_square_error(s_true, s_pred):
    return K.sqrt(K.mean(K.pow(s_true - s_pred, 2)))

model = Model(inputs=x_inp, outputs=score_prediction)
model.compile(
    optimizer=optimizers.Adam(lr=1e-2),
    loss=losses.mean_squared_error,
    metrics=[root_mean_square_error, metrics.mae],
)


# 파헤쳐 보자
# hinsage.input_dims = generator.graph.node_feature_sizes()
# {'movie': 19, 'user': 24}

# generator.schema.type_adjacency_list(generator.head_node_types, len(num_samples))
# = hinsage.subtree_schema

# 차례대로 1~2차 이웃, 1차 이웃을 의미함
# hinsage.neigh_trees[0]
# = [('user', [2]), ('movie', [3]), ('movie', [4]), ('user', [5])]
# hinsage.neigh_tress[1]
# = [('user', [2]), ('movie', [3])]


# {node type: dimension} per layer -- hinsage.dims
# [{'movie': 19, 'user': 24}, {'user': 32, 'movie': 32}, {'user': 32, 'movie': 32}]

# ------
# x_inp: hinsage call 함수의 xin
# out = next(iter(train_gen)) 했을 때 out[0] 하면 그 예시를 볼 수 있음

# hinsage call method, apply_layer method 를 보면
#self.layer_tensors = []
#h_layer = x_inp
#for layer in range(0, 2):
#    h_layer = apply_layer(h_layer, layer)
#    self.layer_tensors.append(h_layer)
# n_layers=2 이므로 위 과정은 총 2번의 loop 가 있는 셈이다. (2-hop)


# 1번째 apply_layer
# for 0, (node_type, neigh_indices) in enumerate(hinsage.neigh_trees[0]):
# 그 안에서 1번째: node_type: 'user', neigh_indices: 2
# head_shape = K.int_shape(x_inp[0])[1] = 1

# 마지막 줄 layer_out.append(self._aggs[layer][node_type]([x_head] + neigh_list))
# hinsage._aggs[layer]




# 이 라이브러리를 그대로 사용하기 위해서는
# G와 edge with ratings 를 어떻게 생성하는지 알아보아야 한다.





