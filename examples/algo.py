# HinSAGE
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K, Input
from tensorflow.keras.layers import Lambda, Dropout, Reshape
from tensorflow.keras.utils import Sequence
from tensorflow.keras import activations, initializers, regularizers, constraints
from typing import List, Callable, Tuple, Dict, Union, AnyStr
import itertools as it
import operator as op
import warnings


# call 메서드
# Input: [(200, 1, 24), (200, 1, 19), (200, 8, 19), (200, 8, 24), (200, 32, 24), (200, 32, 19)]
# Input은 리스트
x_inp = next(iter(train_gen))[0]

# 메인
self = hinsage
# Form HinSAGE layers iteratively
self.layer_tensors = []
h_layer = x_inp

def apply_layer(x: List, layer: int):
    layer_out = []
    for i, (node_type, neigh_indices) in enumerate(self.neigh_trees[layer]):
        # The shape of the head node is used for reshaping the neighbour inputs
        head_shape = K.int_shape(x[i])[1]

        # Apply dropout and reshape neighbours per node per layer
        neigh_list = [
            Dropout(self.dropout)(
                Reshape(
                    (
                        head_shape,
                        self.n_samples[self._depths[i]],
                        self.dims[layer][self.subtree_schema[neigh_index][0]],
                    )
                )(x[neigh_index])
            )
            for neigh_index in neigh_indices
        ]

        # Apply dropout to head inputs
        x_head = Dropout(self.dropout)(x[i])

        # Apply aggregator to head node and reshaped neighbour nodes
        layer_out.append(self._aggs[layer][node_type]([x_head] + neigh_list))

    return layer_out

# apply_layer 파헤치기
# 첫 번째 apply_layer: layer -> 0
x = h_layer
layer_out = []

for i, (node_type, neigh_indices) in enumerate(self.neigh_trees[0]):
    # 0, user, [2] --
    # 1, movie, [3]
    # 2, movie, [4]
    # 3, user, [5]

    # example
    # i, node_type, neigh_indices = 0, 'user', [2]

    # The shape of the head node is used for reshaping the neighbour inputs
    # 차례대로 1, 1, 8, 8 - head node의 수
    head_shape = K.int_shape(x[i])[1]

    # Aplly dropout and reshape neighbours per node per layer
    # 첫 번째 사례를 기준으로
    # head_shape = 1
    # self.n_samples[self._depths[i]] = 8
    # self.dims[0][self.subtree_schema[neigh_index][0]] 에서
    # self.dims = [{'movie': 19, 'user': 24}, {'user': 32, 'movie': 32}, {'user': 32, 'movie': 32}]
    #  -> self.dims[0] = {'movie': 19, 'user': 24}
    # self.subtree_schema[neigh_index][0] = self.subtree_schema[2][0] = 'movie'
    #  -> 19
    # shape 변화: (None, 8, 19) -> (None, 1, 8, 19)
    # input의 종류마다 변수의 수와 이웃의 수가 다르기 때문에 이와 같은 과정이 필요함

    neigh_list = [
        Dropout(self.dropout)(
            Reshape(
                (
                    head_shape,
                    self.n_samples[self._depths[i]],
                    self.dims[0][self.subtree_schema[neigh_index][0]],
                )
            )(x[neigh_index])
        )
        for neigh_index in neigh_indices
    ]

    # Apply dropout to head inputs
    # 첫 번째 사례의 경우 (None, 1, 24)
    x_head = Dropout(self.dropout)(x[i])

    # Apply aggregator to head node and reshaped neighbour nodes
    # node type에 따라 다른 Aggregator를 씀
    # 첫 번째 사례에서 x_head는 User: (None, 1, 24)
    # neigh_list는 Item: (None, 1, 8, 19)
    # 아래에서 local_out: (None, 1, 32) -> User Head Node의 1차 Embedding Matrix
    # -> Item은 i=1에서 만들어진다.
    local_out = self._aggs[0][node_type]([x_head] + neigh_list)
    layer_out.append(local_out)

    # self.aggs를 파보자


# Mean Aggregator
from tensorflow.keras.layers import Layer
HinSAGEAggregator = Layer

class MeanHinAggregator(HinSAGEAggregator):
    def __init__(
        self,
        output_dim: int = 0,
        bias: bool = False,
        act: Union[Callable, AnyStr] = "relu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs,
    ):
        self.output_dim = output_dim
        if output_dim % 2 != 0:
            raise ValueError("The output_dim must be a multiple of two.")
        self.half_output_dim = output_dim // 2
        self.has_bias = bias
        self.act = activations.get(act)
        self.nr = None
        self.w_neigh = []
        self.w_self = None
        self.bias = None

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        super().__init__(**kwargs)

    def get_config(self):
        config = {
            "output_dim": self.output_dim,
            "bias": self.has_bias,
            "act": activations.serialize(self.act),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def build(self, input_shape):
        # Args: input_shape (list of list of int): Shape of input per neighbour type.
        # nr = 1

        # Weight matrix for each type of neighbour
        # If there are no neighbours (input_shape[x][2]) for an input
        # then do not create weights as they are not used.
        # 아래에서 r = 0이므로
        # shape = ( , )에서 input_shape[1][3]의 의미는,
        # input_shape[1] -> neigh_list의 input shape 이기 때문에
        # (None, 1, 8, 32) 혹은 (None, 1, 8, 19)와 같은 식이 므로 마지막 차원, 즉 변수 수를 의미함
        # 따라서 w_neigh의 shape = (32, 16) 혹은 (32, 19) 혹은 (32, 24) 와 같은 형식임
        self.nr = len(input_shape) - 1
        self.w_neigh = [
            self.add_weight(
                name="w_neigh_" + str(r),
                shape=(int(input_shape[1 + r][3]), self.half_output_dim),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )
            if input_shape[1 + r][2] > 0
            else None
            for r in range(self.nr)
        ]

        # Weight matrix for self
        # input_shape[0]은 x_head의 shape을 의미하므로 이 x_head는
        # (200, 8, 32/19/24)의 형식이므로
        # w_self의 shape = (32, 16) 혹은 32, 19) 혹은 (32, 24) 와 같은 형식임
        self.w_self = self.add_weight(
            name="w_self",
            shape=(int(input_shape[0][2]), self.half_output_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

        # Optional bias
        if self.has_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.output_dim],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )

        super().build(input_shape)

    def call(self, x, **kwargs):
        """
        Apply MeanAggregation on input tensors, x

        Args:
          x: List of Keras Tensors with the following elements

            - x[0]: tensor of self features shape (n_batch, n_head, n_feat)
            - x[1+r]: tensors of neighbour features each of shape (n_batch, n_head, n_neighbour[r], n_feat[r])

        예시:
        x = [x_head] + neigh_list
        x[0] = x_head -> (200, 1, 24), User
        x[1+r] = neigh_list -> (200, 1, 8, 19), Item

        Returns:
            Keras Tensor representing the aggregated embeddings in the input.

        """
        # Calculate the mean vectors over the neigbours of each relation (edge) type
        neigh_agg_by_relation = []
        # nr = 0 이기 때문에 r = 0만 적용됨
        for r in range(self.nr):
            # The neighbour input tensors for relation r
            # z: neigh_list
            z = x[1 + r]

            # If there are neighbours aggregate over them
            # 사실상 0보다 클 수밖에 없음
            # K.mean(z, axis=2): z (200, 1, 8, 19) --> (200, 1, 19)
            # (200, 1, 19) * (19, 16)
            # --> z_agg: (200, 1, 16)
            if z.shape[2] > 0:
                z_agg = K.dot(K.mean(z, axis=2), self.w_neigh[r])

            # Otherwise add a synthetic zero vector
            else:
                z_shape = K.shape(z)
                w_shape = self.half_output_dim
                z_agg = tf.zeros((z_shape[0], z_shape[1], w_shape))

            neigh_agg_by_relation.append(z_agg)

        # Calculate the self vector shape (n_batch, n_head, n_out_self)
        # (200, 1, 16) = (200, 1, 24) * (24, 16)
        from_self = K.dot(x[0], self.w_self)

        # Sum the contributions from all neighbour averages shape (n_batch, n_head, n_out_neigh)
        from_neigh = sum(neigh_agg_by_relation) / self.nr

        # Concatenate self + neighbour features, shape (n_batch, n_head, n_out)
        # (200, 1, 32)
        total = K.concatenate(
            [from_self, from_neigh], axis=2
        )  # YT: this corresponds to concat=Partial
        # TODO: implement concat=Full and concat=False

        return self.act((total + self.bias) if self.has_bias else total)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.
        Assumes that the layer will be built to match that input shape provided.

        Args:
            input_shape (tuple of int)
                Shape tuples can include `None` for free dimensions, instead of an integer.

        Returns:
            An input shape tuple.
        """
        return input_shape[0][0], input_shape[0][1], self.output_dim


config = self._aggs[0]['user'].get_config()
nr = self._aggs[0]['user'].nr
w_neigh = self._aggs[0]['user'].w_neigh[0] # w_neigh가 리스트로 되어 있어서
w_self = self._aggs[0]['user'].w_self
print(nr, w_neigh.shape, w_self.shape) # 1, (19, 16), (24, 16)
# 16인 이유는 half_output_dim = 32/2 이기 때문, hidden_size라고 생각하면 됨

# Input: [x_head] + neigh_list =  (None, 1, 24), (None, 1, 8, 19)
# input_shape = [TensorShape([200, 1, 24]), TensorShape([200, 1, 8, 19])]
# output_shape = self._aggs[0]['user'].compute_output_shape(input_shape)
# --> (200, 1, 32)


# 두 번째 apply_layer: layer -> 1
layer = 1
out0 = apply_layer(h_layer, 0)
x = out0
layer_out = []

for i, (node_type, neigh_indices) in enumerate(self.neigh_trees[1]):
    # 0, user, [2]
    # 1, movie, [3]

    # example
    # i, node_type, neigh_indices = 0, 'user', [2]

    # The shape of the head node is used for reshaping the neighbour inputs
    # 차례대로 1, 1, 8, 8 - head node의 수
    head_shape = K.int_shape(x[i])[1]

    # Apply dropout and reshape neighbours per node per layer
    # 첫 번째 사례를 기준으로
    # head_shape = 1
    # self.n_samples[self._depths[i]] = 8
    # self.dims[1][self.subtree_schema[neigh_index][0]] 에서
    # self.dims = [{'movie': 19, 'user': 24}, {'user': 32, 'movie': 32}, {'user': 32, 'movie': 32}]
    #  -> self.dims[1] = {'movie': 32, 'user': 232}
    # self.subtree_schema[neigh_index][0] = self.subtree_schema[2][0] = 'movie'
    #  -> 32
    # shape 변화: (None, 8, 32) -> (None, 1, 8, 32)
    # input의 종류마다 변수의 수와 이웃의 수가 다르기 때문에 이와 같은 과정이 필요함

    # neigh_index는 차례대로 2, 3이다. (layer=0 일 때는 0, 1)
    # x[2]: (None, 8, 32), x[3]: (None, 8, 32)
    # x[2] -> (None, 1, 8, 32)
    neigh_list = [
        Dropout(self.dropout)(
            Reshape(
                (
                    head_shape,
                    self.n_samples[self._depths[i]],
                    self.dims[1][self.subtree_schema[neigh_index][0]],
                )
            )(x[neigh_index])
        )
        for neigh_index in neigh_indices
    ]

    # Apply dropout to head inputs
    # 첫 번째 사례의 경우 (None, 1, 24)
    x_head = Dropout(self.dropout)(x[i])

    # Apply aggregator to head node and reshaped neighbour nodes
    # node type에 따라 다른 Aggregator를 씀
    # 첫 번째 사례에서 x_head는 User: (None, 1, 24)
    # neigh_list는 Item: (None, 1, 8, 19)
    # 아래에서 local_out: (None, 1, 32) -> User Head Node의 1차 Embedding Matrix
    # -> Item은 i=1에서 만들어진다.
    local_out = self._aggs[1][node_type]([x_head] + neigh_list)
    layer_out.append(local_out)

# self._aggs[0]
# {'user': <stellargraph.layer.hinsage.MeanHinAggregator object at 0x00000243A9CF0220>,
#  'movie': <stellargraph.layer.hinsage.MeanHinAggregator object at 0x00000243A9CF06A0>}
# self._aggs[1]
# {'user': <stellargraph.layer.hinsage.MeanHinAggregator object at 0x00000243A9CF0AC0>,
#  'movie': <stellargraph.layer.hinsage.MeanHinAggregator object at 0x00000243A9CF0EE0>}



# layer = 0
out0 = apply_layer(h_layer, 0)
self.layer_tensors.append(out0)

print([out0[i].shape for i in range(0, 4)])
# 1차 이웃의 Embedding 벡터
# (200, 1, 32), (200, 1, 32), (200, 8, 32), (200, 8, 32)

# layer = 1
out1 = apply_layer(out0, 1)
self.layer_tensors.append(out1)

print([out1[i].shape for i in range(0, 2)])
# Target Node 200개의 Embedding 벡터
# (200, 1, 32), (200, 1, 32)

# len(self.layer_tensors[0]), len(self.layer_tensors[1])
# (4, 2) -> 1차 이웃 임베딩 벡터, Target Node 임베딩 벡터

# Remove neighbourhood dimension from output tensors
# note that at this point h_layer contains the output tensor of the top (last applied) layer of the stack
final_layer = [
    Reshape(K.int_shape(x)[2:])(x) for x in out1 if K.int_shape(x)[1] == 1
]

# final_layer: [(200, 32), (200, 32)]


# Return final layer output tensor with optional normalization
final_output = [self._normalization(xi) for xi in final_layer]







