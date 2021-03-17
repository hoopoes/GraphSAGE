# Train PairSAGE

from sklearn.model_selection import train_test_split

import stellargraph as sg
from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import link_regression
from models import PairSAGE
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

edges_train, edges_test = train_test_split(
    edges_with_ratings, train_size=train_size, test_size=test_size)

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


# Model
pairsage = PairSAGE(layer_sizes=[32, 32], generator=generator)

x_inp, x_out = pairsage.in_out_tensors()
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

num_workers=4

test_metrics = model.evaluate(
    test_gen, verbose=1, use_multiprocessing=False, workers=num_workers)

print("Untrained model's Test Evaluation:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))


history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=epochs,
    verbose=1,
    shuffle=False,
    use_multiprocessing=False,
    workers=num_workers)

sg.utils.plot_history(history)