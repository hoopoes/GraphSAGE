# GraphSAGE Implementation for Bipartite User-Item pair graph
# got helped from https://github.com/stellargraph/stellargraph

import itertools as it
import operator as op
from typing import List, Callable, Tuple, Dict, Union, AnyStr

from tensorflow.keras import backend as K, Input
from tensorflow.keras.layers import Lambda, Dropout, Reshape

from utils import Aggregator


class GraphSAGE:
    def __init__(self, layer_sizes, generator=None,
        agg_option='mean', bias=True, dropout=0.0, normalize="l2"):
        """
        Graph SAGE Model with User-Item Settings

        :param layer_sizes (list): hidden_size of layers
        :param generator: PairSAGEGenerator object
        :param agg_option (str): 'mean' or 'max'
        :param bias (bool): Bias Option
        :param dropout (float): Dropout Rate
        :param normalize (String or None): Normalization Option
        """
        # Aggregator & Normalization
        self._aggregator = Aggregator
        self.agg_option = agg_option

        if normalize == "l2":
            self._normalization = Lambda(lambda x: K.l2_normalize(x, axis=-1))
        else:
            self._normalization = Lambda(lambda x: x)

        # Get the sampling tree, input_dim, and num_samples from the generator
        if generator is not None:
            self._get_sizes_from_generator(generator)
        else:
            raise ValueError("Generator must be inserted")

        # Set parameters for the model
        self.n_layers = len(self.n_samples)
        self.bias = bias
        self.dropout = dropout

        # Neighborhood info per layer
        self.neigh_trees = self._eval_neigh_tree_per_layer(
            [li for li in self.subtree_schema if len(li[1]) > 0]
        )

        # Depth of each input tensor i.e. number of hops from root nodes
        self._depths = [
            self.n_layers
            + 1
            - sum([1 for li in [self.subtree_schema] + self.neigh_trees if i < len(li)])
            for i in range(len(self.subtree_schema))
        ]

        # Dict of {node type: dimension} per layer
        self.dims = [
            dim
            if isinstance(dim, dict)
            else {k: dim for k, _ in ([self.subtree_schema] + self.neigh_trees)[layer]}
            for layer, dim in enumerate([self.input_dims] + layer_sizes)
        ]

        # Activation function for each layer
        self.activations = ["relu"] * (self.n_layers - 1) + ["linear"]

        # Aggregator functions for each layer
        # output_dim: 32
        # self.n_layers = 2, node_type 은 user, item 이렇게 2개 이므로
        # --> Aggregator 는 총 4개임
        # 아래에서 layer = 0, 1 두 경우 뿐임
        # layer=0 -> self.dims[1].items() = [('user', 32), ('item', 32)])
        # layer=1 -> self.dims[2].items() = [('user', 32), ('item', 32)])
        # 따라서 결국 output_dim = 32로 통일된다.
        self._aggs = [
            {
                node_type: self._aggregator(
                    output_dim,
                    has_bias=self.bias,
                    activation_function=self.activations[layer],
                    agg_option=self.agg_option
                )
                for node_type, output_dim in self.dims[layer + 1].items()
            }
            for layer in range(self.n_layers)
        ]

    def __repr__(self):
        identity = "<GraphSAGE model with {} layers>".format(self.n_layers)
        return identity

    def _get_sizes_from_generator(self, generator):
        self.input_dims = generator.graph.node_feature_sizes    # dict
        self.n_samples = generator.num_samples
        self.subtree_schema = generator.type_adjacency_list
        #self.subtree_schema = generator.type_adjacency_list(
        #    generator.head_node_types, len(self.n_samples))

    @staticmethod
    def _eval_neigh_tree_per_layer(input_tree):
        """
        Function to evaluate the neighbourhood tree structure for every layer. The tree
        structure at each layer is a truncated version of the previous layer.

        Args:
            input_tree: Neighbourhood tree for the input batch

        Returns:
            List of neighbourhood trees

        """
        reduced = [
            li
            for li in input_tree
            if all(li_neigh < len(input_tree) for li_neigh in li[1])
        ]
        return (
            [input_tree]
            if len(reduced) == 0
            else [input_tree] + GraphSAGE._eval_neigh_tree_per_layer(reduced)
        )

    def __call__(self, xin: List):
        """
        Apply aggregator layers
        Args: x (list of Tensor): Batch input features
        Returns: Output tensor
        """

        def apply_layer(x: List, layer: int):
            """
            Compute the list of output tensors for a single SAGE layer

            Args:
                x (List[Tensor]): Inputs to the layer
                layer (int): Layer index

            Returns:
                Outputs of applying the aggregators as a list of Tensors
            """
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

        # Form HinSAGE layers iteratively
        self.layer_tensors = []
        h_layer = xin
        for layer in range(0, self.n_layers):
            h_layer = apply_layer(h_layer, layer)
            self.layer_tensors.append(h_layer)

        # Remove neighbourhood dimension from output tensors
        # note that at this point h_layer contains the output tensor of the top (last applied) layer of the stack
        h_layer = [
            Reshape(K.int_shape(x)[2:])(x) for x in h_layer if K.int_shape(x)[1] == 1
        ]

        # Return final layer output tensor with optional normalization
        return (
            self._normalization(h_layer[0])
            if len(h_layer) == 1
            else [self._normalization(xi) for xi in h_layer]
        )

    def _input_shapes(self) -> List[Tuple[int, int]]:
        """
        Returns the input shapes for the tensors of the supplied neighbourhood type tree

        Returns:
            A list of tuples giving the shape (number of nodes, feature size) for
            the corresponding item in the neighbourhood type tree (self.subtree_schema)
        """
        neighbor_sizes = list(it.accumulate([1] + self.n_samples, op.mul))

        def get_shape(stree, cnode, level=0):
            adj = stree[cnode][1]
            size_dict = {
                cnode: (neighbor_sizes[level], self.input_dims[stree[cnode][0]])
            }
            if len(adj) > 0:
                size_dict.update(
                    {
                        k: s
                        for a in adj
                        for k, s in get_shape(stree, a, level + 1).items()
                    }
                )
            return size_dict

        input_shapes = dict()
        for ii in range(len(self.subtree_schema)):
            input_shapes_ii = get_shape(self.subtree_schema, ii)
            # Update input_shapes if input_shapes_ii.keys() are not already in input_shapes.keys():
            if (
                len(set(input_shapes_ii.keys()).intersection(set(input_shapes.keys())))
                == 0
            ):
                input_shapes.update(input_shapes_ii)

        return [input_shapes[ii] for ii in range(len(self.subtree_schema))]

    def in_out_tensors(self):
        """
        Builds a HinSAGE model for node or link/node pair prediction, depending on the generator used to construct
        the model (whether it is a node or link/node pair generator).

        Returns:
            tuple: ``(x_inp, x_out)``, where ``x_inp`` is a list of Keras input tensors
                for the specified HinSAGE model (either node or link/node pair model) and ``x_out`` contains
                model output tensor(s) of shape (batch_size, layer_sizes[-1]).
        """

        # Create tensor inputs
        x_inp = [Input(shape=s) for s in self._input_shapes()]

        # Output from HinSAGE model
        x_out = self(x_inp)
        return x_inp, x_out

