# Utility Functions for PairSAGE
# PairSAGE: GraphSAGE Implementation for Bipartite User-Item pair graph
# got helped from https://github.com/stellargraph/stellargraph

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import activations

import collections
from collections import namedtuple
from typing import List, Callable, Tuple, Dict, Union, AnyStr
import random as rn
import numpy as np
import numpy.random as np_rn

# 1) random_state, is_real_iterable
RandomState = namedtuple("RandomState", "random, numpy")

def _global_state():
    return RandomState(rn, np_rn)

def _seeded_state(s):
    return RandomState(rn.Random(s), np_rn.RandomState(s))

_rs = _global_state()

def random_state(seed):
    if seed is None:
        return _rs
    else:
        return _seeded_state(seed)

def is_real_iterable(x):
    # 문자열, 바이트가 아니면서 순회 가능한 객체인지 확인
    return isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes))

# 2) Aggregator
class Aggregator(Layer):
    def __init__(
            self,
            output_dim: int = 32,
            has_bias: bool = True,
            activation_function: Union[Callable, AnyStr] = "relu",
            agg_option = 'mean',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        if output_dim % 2 != 0:
            raise ValueError("The output_dim must be a multiple of two.")
        self.half_output_dim = output_dim // 2
        self.has_bias = has_bias
        self.activations = activations.get(activation_function)
        self.agg_option = agg_option
        self.w_neigh = None
        self.w_self = None
        self.bias = None
        assert self.agg_option in ['mean', 'max'], "agg_option must be mean or max"

        super().__init__(**kwargs)

    def __repr__(self):
        name = "Mean Aggregator with output_dim {}".format(self.output_dim)
        return name

    def build(self, input_shape):
        # input_shape: shape of input per neighbor type
        # input: x = [x_head] + neigh_list
        #  -> [[None, 1, n_user_feature], [None, 1, n_neigh, n_item_feature]]
        #  -> [[None, 1, 24], [None, 1, 8, 19]]

        # Define Weight Matrix for each type of neighbor
        # w_neigh: (19, 16), w_self: (24, 16)
        self.w_neigh = self.add_weight(
            name="w_neigh",
            shape=(int(input_shape[1][3]), self.half_output_dim),
            initializer="glorot_uniform",
            trainable=True)

        self.w_self = self.add_weight(
            name="w_self",
            shape=(int(input_shape[0][2]), self.half_output_dim),
            initializer="glorot_uniform",
            trainable=True)

        if self.has_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.output_dim],
                initializer="zeros",
                trainable=True)

        super().build(input_shape)

    def call(self, x, **kwargs):
        """
        :param x: list of input tensors
         - x[0]: self features (None, n_head, n_features)
         - x[1]: neighbor features (None, n_head, n_neigh, n_features)
         ex) x[0]: (200, 1, 24), x[1]: (200, 1, 8, 19)
        :return: aggregated embedding matrices corresponding to the input
        """

        # 1) Aggregation over Neighbor
        x_neigh = x[1]

        # Example
        # K.mean: (None, 1, 8, 19) --> (None, 1, 19)
        # (None, 1, 19) * (19, 16) --> x_neigh_agg: (None, 1, 16)
        if self.agg_option == 'mean':
            neigh_agg = tf.matmul(K.mean(x_neigh, axis=2), self.w_neigh)
        elif self.agg_option == 'max':
            neigh_agg = tf.matmul(K.max(x_neigh, axis=2), self.w_neigh)

        # 2) Self Aggregation
        self_agg = tf.matmul(x[0], self.w_self)    # (None, n_head, half_output_dim)

        # 3) Concatenation self & neighbor aggregation: (None, n_head, output_dim)
        aggregated_vectors = tf.concat([self_agg, neigh_agg], axis=2)
        aggregated_vectors = self.activations(
            (aggregated_vectors + self.bias) if self.has_bias else aggregated_vectors)
        return aggregated_vectors


# used in defining StellarGraph
def extract_element_features(element_data, unique, name, ids, type, use_ilocs):
    if ids is None:
        if type is None:
            type = unique(
                f"{name}_type: in a non-homogeneous graph, expected a {name} type and/or '{name}s' to be passed; found neither '{name}_type' nor '{name}s', and the graph has {name} types: %(found)s"
            )

        return element_data.features_of_type(type)

    ids = np.asarray(ids)

    if len(ids) == 0:
        # empty lists are cast to a default array type of float64 -
        # must manually specify integer type if empty, in which case we can pretend we received ilocs
        ilocs = ids.astype(dtype=np.uint8)
        use_ilocs = True
    elif use_ilocs:
        ilocs = ids
    else:
        ilocs = element_data.ids.to_iloc(ids)

    valid = element_data.ids.is_valid(ilocs)
    all_valid = valid.all()
    valid_ilocs = ilocs if all_valid else ilocs[valid]

    if type is None:
        try:
            # no inference required in a homogeneous-node graph
            type = unique()
        except ValueError:
            # infer the type based on the valid nodes
            types = np.unique(element_data.type_of_iloc(valid_ilocs))

            if len(types) == 0:
                raise ValueError(
                    f"must have at least one node for inference, if `{name}_type` is not specified"
                )
            if len(types) > 1:
                raise ValueError(f"all {name}s must have the same type")

            type = types[0]

    if all_valid:
        return element_data.features(type, valid_ilocs)

    # If there's some invalid values, they get replaced by zeros; this is designed to allow
    # models that build fixed-size structures (e.g. GraphSAGE) based on neighbours to fill out
    # missing neighbours with zeros automatically, using None as a sentinel.

    # FIXME: None as a sentinel forces nodes to have dtype=object even with integer IDs, could
    # instead use an impossible integer (e.g. 2**64 - 1)

    # everything that's not the sentinel should be valid
    if not use_ilocs:
        non_nones = ids != None
        element_data.ids.require_valid(ids[non_nones], ilocs[non_nones])

    sampled = element_data.features(type, valid_ilocs)
    features = np.zeros((len(ids), sampled.shape[1]))
    features[valid] = sampled

    return features



