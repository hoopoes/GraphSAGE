# Train PairSAGE
from stellargraph import datasets
from sklearn.model_selection import train_test_split

#from stellargraph.mapper import HinSAGELinkGenerator
#from stellargraph.layer import HinSAGE, link_regression
from tensorflow.keras import Model, optimizers, losses, metrics
import tensorflow.keras.backend as K


batch_size = 200
epochs = 20
train_size = 0.7
test_size = 0.3

dataset = datasets.MovieLens()
G, edges_with_ratings = dataset.load()

edges_train, edges_test = train_test_split(
    edges_with_ratings, train_size=train_size, test_size=test_size)

# user-item edge 리스트
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

