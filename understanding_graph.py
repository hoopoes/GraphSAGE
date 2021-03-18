# Understanding StellarGraph Class

import numpy as np
import pandas as pd
from stellargraph.core.graph import StellarGraph
from stellargraph.core.indexed_array import IndexedArray

edges = pd.DataFrame(
    {"source": ["a", "b", "c", "d", "a"], "target": ["b", "c", "d", "a", "c"]}
)

graph = StellarGraph(edges=edges)

# 1) Feature 추가 방법
# As a IndexedArray (no column names):
feature_array = np.array([[-1, 0.4], [2, 0.1], [-3, 0.9], [4, 0]])
nodes = IndexedArray(feature_array, index=["a", "b", "c", "d"])

# As a Pandas DataFrame:
nodes = pd.DataFrame(
    {"x": [-1, 2, -3, 4], "y": [0.4, 0.1, 0.9, 0]}, index=["a", "b", "c", "d"]
)

# As a NumPy array:
# Note, edges must change to using 0, 1, 2, 3 (instead of a, b, c, d)
nodes = feature_array


# 2) Heterogeneous Graph
edges = pd.DataFrame({
    "source": ["a", "a", "a", "b", "b"],
    "target": ["c", "d", "e", "c", "d"]
})

users = IndexedArray(np.array([[-1], [4]]), index=["a", "b"])
items = IndexedArray(
    np.array([[0.4, 100], [0.1, 200], [0.9, 300]]), index=["c", "d", "e"])

graph = StellarGraph({"users": users, "items": items}, edges)
print(graph.info())

#--------
# 3) Modified StellarGraph Class

from typing import Iterable, Any, Mapping, List, Optional, Set
from collections import defaultdict, namedtuple
import pandas as pd
import numpy as np
import scipy.sparse as sps
import warnings

from utils import is_real_iterable, extract_element_features

from stellargraph import globalvar
from stellargraph.core.schema import GraphSchema, EdgeType
from stellargraph.core.element_data import NodeData, EdgeData, ExternalIdIndex
from stellargraph.core.validation import comma_sep, separated
from stellargraph.core import convert


class StellarGraph:
    def __init__(
        self,
        nodes=None,
        edges=None,
        *,
        is_directed=False,
        source_column=globalvar.SOURCE,
        target_column=globalvar.TARGET,
        edge_weight_column=globalvar.WEIGHT,
        edge_type_column=None,
        node_type_default=globalvar.NODE_TYPE_DEFAULT,
        edge_type_default=globalvar.EDGE_TYPE_DEFAULT,
        dtype="float32"
    ):

        # infer_nodes_from_edges: 이게 필요한가?
        if nodes is None:
            nodes_after_inference = self._infer_nodes_from_edges(
                edges, source_column, target_column
            )
            nodes = pd.DataFrame([], index=nodes_after_inference)

        if edges is None:
            edges = {}

        self._is_directed = is_directed

        nodes_is_internal = isinstance(nodes, NodeData)
        edges_is_internal = isinstance(edges, EdgeData)
        any_internal = nodes_is_internal or edges_is_internal

        if not any_internal:
            internal_nodes = convert.convert_nodes(
                nodes, name="nodes", default_type=node_type_default, dtype=dtype,
            )

            internal_edges = convert.convert_edges(
                edges,
                name="edges",
                default_type=edge_type_default,
                source_column=source_column,
                target_column=target_column,
                weight_column=edge_weight_column,
                type_column=edge_type_column,
                nodes=internal_nodes,
                dtype=dtype,
            )
        else:
            if not edges_is_internal:
                raise TypeError(
                    f"edges: expected type 'EdgeData' when 'nodes' has type 'NodeData', found {type(edges).__name__}"
                )
            if not nodes_is_internal:
                raise TypeError(
                    f"nodes: expected type 'NodeData' when 'edges' has type 'EdgeData', found {type(nodes).__name__}"
                )

            params = locals()
            for param, expected in self.__init__.__kwdefaults__.items():
                if param == "is_directed":
                    continue

                if params[param] is not expected:
                    raise ValueError(
                        f"{param}: expected the default value ({expected!r}) when constructing from 'NodeData' and 'EdgeData', found {params[param]!r}. (All parameters except 'nodes', 'edges' and 'is_directed' must be left unset.)"
                    )

            internal_nodes = nodes
            internal_edges = edges

            # FIXME: it would be good to do more validation that 'nodes' and 'edges' match here

        self._nodes = internal_nodes
        self._edges = internal_edges

    @staticmethod
    def _infer_nodes_from_edges(edges, source_column, target_column):
        # `convert_edges` nicely flags any errors in edges; inference here is lax rather than duplicate that
        if isinstance(edges, dict):
            dataframes = edges.values()
        else:
            dataframes = [edges]

        found_columns = [
            type_edges[column]
            for type_edges in dataframes
            if isinstance(type_edges, pd.DataFrame)
            for column in [source_column, target_column]
            if column in type_edges.columns
        ]

        if found_columns:
            return pd.unique(np.concatenate(found_columns))

        return []


    # customise how a missing attribute is handled to give better error messages for the NetworkX
    # -> no NetworkX transition.
    def __getattr__(self, item):
        import networkx

        try:
            # do the normal access, in case the attribute actually exists, and to get the native
            # python wording of the error
            return super().__getattribute__(item)
        except AttributeError as e:
            if hasattr(networkx.MultiDiGraph, item):
                # a networkx class has this as an attribute, so let's assume that it's old code
                # from before the conversion and replace (the `from None`) the default exception
                # with one with a more specific message that guides the user to the fix
                type_name = type(self).__name__
                raise AttributeError(
                    f"{e.args[0]}. The '{type_name}' type no longer inherits from NetworkX types: use a new StellarGraph method, or, if that is not possible, the `.to_networkx()` conversion function."
                ) from None

            # doesn't look like a NetworkX method so use the default error
            raise

    def is_directed(self) -> bool:
        return self._is_directed

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def number_of_edges(self) -> int:
        return len(self._edges)

    def nodes(self, node_type=None, use_ilocs=False) -> Iterable[Any]:
        """
        Obtains the collection of nodes in the graph.

        Args:
            node_type (hashable, optional): a type of nodes that exist in the graph
            use_ilocs (bool): if True return :ref:`node ilocs <iloc-explanation>` as a ``range`` object

        Returns:
            All the nodes in the graph if ``node_type`` is ``None``, otherwise all the nodes in the
            graph of type ``node_type``.
        """
        if node_type is None:
            all_ids = self._nodes.ids.pandas_index
            if use_ilocs:
                return range(self.number_of_nodes())
            return all_ids

        ilocs = self._nodes.type_range(node_type)
        if use_ilocs:
            return ilocs
        return self._nodes.ids.from_iloc(ilocs)

    def _to_edges(self, edge_arrs):
        edges = list(zip(*(arr for arr in edge_arrs[:3] if arr is not None)))
        if edge_arrs[3] is not None:
            return edges, edge_arrs[3]
        return edges

    def edges(
        self, include_edge_type=False, include_edge_weight=False, use_ilocs=False
    ) -> Iterable[Any]:
        """
        Obtains the collection of edges in the graph.

        Args:
            include_edge_type (bool):
                A flag that indicates whether to return edge types of format (node 1, node 2, edge
                type) or edge pairs of format (node 1, node 2).
            include_edge_weight (bool):
                A flag that indicates whether to return edge weights.  Weights are returned in a
                separate list.
            use_ilocs (bool): if True return :ref:`ilocs for nodes (and edge types) <iloc-explanation>`

        Returns:
            The graph edges. If edge weights are included then a tuple of (edges, weights).
        """
        edge_arrs = self.edge_arrays(
            include_edge_type, include_edge_weight, use_ilocs=use_ilocs
        )
        return self._to_edges(edge_arrs)

    def edge_arrays(
        self, include_edge_type=False, include_edge_weight=False, use_ilocs=False
    ) -> tuple:
        """
        Obtains the collection of edges in the graph as a tuple of arrays (sources, targets, types, weights).
        ``types`` and ``weights`` will be `None` if the optional parameters are not specified.

        Args:
            include_edge_type (bool): A flag that indicates whether to return edge types.
            include_edge_weight (bool): A flag that indicates whether to return edge weights.
            use_ilocs (bool): if True return :ref:`ilocs for nodes (and edge types) <iloc-explanation>`

        Returns:
            A tuple containing 1D arrays of the source and target nodes (sources, targets, types, weights).
            Setting include_edge_type and/or include_edge_weight to True will include arrays of edge types
            and/or edge weights in this tuple, otherwise they will be set to ``None``.
        """
        types = types = self._edges.type_ilocs if include_edge_type else None
        weights = self._edges.weights if include_edge_weight else None
        sources = self._edges.sources
        targets = self._edges.targets

        if not use_ilocs:
            sources = self.node_ilocs_to_ids(sources)
            targets = self.node_ilocs_to_ids(targets)
            types = self._edges.type_of_iloc(slice(None)) if include_edge_type else None
        return sources, targets, types, weights

    def has_node(self, node: Any) -> bool:
        return node in self._nodes

    def _transform_edges(
        self, other_node, ilocs, include_edge_weight, filter_edge_types, use_ilocs
    ):
        if include_edge_weight:
            weights = self._edges.weights[ilocs]
        else:
            weights = None

        if not use_ilocs:
            other_node = self._nodes.ids.from_iloc(other_node)

        if filter_edge_types is not None:
            if not use_ilocs:
                filter_edge_types = self._edges.types.to_iloc(filter_edge_types)
            edge_type_ilocs = self._edges.type_ilocs[ilocs]
            correct_type = np.isin(edge_type_ilocs, filter_edge_types)

            other_node = other_node[correct_type]
            if weights is not None:
                weights = weights[correct_type]

        if weights is not None:
            return other_node, weights

        return other_node

    def _to_neighbors(self, neigh_arrs, include_edge_weight):
        if include_edge_weight:
            return [
                NeighbourWithWeight(neigh, weight) for neigh, weight in zip(*neigh_arrs)
            ]
        return list(neigh_arrs)

    def neighbor_arrays(
        self, node: Any, include_edge_weight=False, edge_types=None, use_ilocs=False
    ):
        """
        Obtains the collection of neighbouring nodes connected to the given node
        as an array of node_ids. If `include_edge_weight` edge is `True` then
        an array of edges weights is also returned in a tuple of `(neighbor_ids, edge_weights)`.

        Args:
            node (any): The node in question.
            include_edge_weight (bool, default False): If True an array of edge weights is also returned.
            edge_types (list of hashable, optional): If provided, only traverse the graph
                via the provided edge types when collecting neighbours.
            use_ilocs (bool): if True `node` is treated as a :ref:`node iloc <iloc-explanation>`
                (and similarly `edge_types` is treated as a edge type ilocs) and the ilocs of each
                neighbour is returned.

        Returns:
            A numpy array of the neighboring nodes. If `include_edge_weight` is `True` then an array
            of edge weights is also returned in a tuple `(neighbor_array, edge_weight_array)`
        """
        if not use_ilocs:
            node = self._nodes.ids.to_iloc([node])[0]

        edge_ilocs = self._edges.edge_ilocs(node, ins=True, outs=True)
        source = self._edges.sources[edge_ilocs]
        target = self._edges.targets[edge_ilocs]
        other_node = np.where(source == node, target, source)

        return self._transform_edges(
            other_node, edge_ilocs, include_edge_weight, edge_types, use_ilocs
        )

    def neighbors(
        self, node: Any, include_edge_weight=False, edge_types=None, use_ilocs=False
    ) -> Iterable[any]:
        """
        Obtains the collection of neighbouring nodes connected
        to the given node.

        Args:
            node (any): The node in question.
            include_edge_weight (bool, default False): If True, each neighbour in the
                output is a named tuple with fields `node` (the node ID) and `weight` (the edge weight)
            edge_types (list of hashable, optional): If provided, only traverse the graph
                via the provided edge types when collecting neighbours.
            use_ilocs (bool): if True `node` is treated as a :ref:`node iloc <iloc-explanation>`
                (and similarly `edge_types` is treated as a edge type ilocs) and the ilocs of each
                neighbour is returned.

        Returns:
            iterable: The neighboring nodes.
        """
        neigh_arrs = self.neighbor_arrays(
            node, include_edge_weight, edge_types, use_ilocs=use_ilocs
        )
        return self._to_neighbors(neigh_arrs, include_edge_weight)

    def in_node_arrays(
        self, node: Any, include_edge_weight=False, edge_types=None, use_ilocs=False
    ):
        """
        Obtains the collection of neighbouring nodes with edges
        directed to the given node. For an undirected graph,
        neighbours are treated as both in-nodes and out-nodes.

        Args:
            node (any): The node in question.
            include_edge_weight (bool, default False): If True an array of edge weights is also returned.
            edge_types (list of hashable, optional): If provided, only traverse the graph
                via the provided edge types when collecting neighbours.
            use_ilocs (bool): if True `node` is treated as a :ref:`node iloc <iloc-explanation>`
                (and similarly `edge_types` is treated as a edge type ilocs) and the ilocs of each
                neighbour is returned.

        Returns:
            A numpy array of the neighboring in-nodes. If `include_edge_weight` is `True` then an array
            of edge weights is also returned in a tuple `(neighbor_array, edge_weight_array)`
        """
        if not self.is_directed():
            # all edges are both incoming and outgoing for undirected graphs
            return self.neighbor_arrays(
                node,
                include_edge_weight=include_edge_weight,
                edge_types=edge_types,
                use_ilocs=use_ilocs,
            )

        if not use_ilocs:
            node = self._nodes.ids.to_iloc([node])[0]
        edge_ilocs = self._edges.edge_ilocs(node, ins=True, outs=False)
        source = self._edges.sources[edge_ilocs]

        return self._transform_edges(
            source, edge_ilocs, include_edge_weight, edge_types, use_ilocs
        )

    def in_nodes(
        self, node: Any, include_edge_weight=False, edge_types=None, use_ilocs=False
    ) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes with edges
        directed to the given node. For an undirected graph,
        neighbours are treated as both in-nodes and out-nodes.

        Args:
            node (any): The node in question.
            include_edge_weight (bool, default False): If True, each neighbour in the
                output is a named tuple with fields `node` (the node ID) and `weight` (the edge weight)
            edge_types (list of hashable, optional): If provided, only traverse the graph
                via the provided edge types when collecting neighbours.
            use_ilocs (bool): if True `node` is treated as a :ref:`node iloc <iloc-explanation>`
                (and similarly `edge_types` is treated as a edge type ilocs) and the ilocs of each
                neighbour is returned.

        Returns:
            iterable: The neighbouring in-nodes.
        """
        neigh_arrs = self.in_node_arrays(
            node, include_edge_weight, edge_types, use_ilocs=use_ilocs
        )
        return self._to_neighbors(neigh_arrs, include_edge_weight)

    def out_node_arrays(
        self, node: Any, include_edge_weight=False, edge_types=None, use_ilocs=False
    ):
        """
        Obtains the collection of neighbouring nodes with edges
        directed from the given node. For an undirected graph,
        neighbours are treated as both in-nodes and out-nodes.

        Args:
            node (any): The node in question.
            include_edge_weight (bool, default False): If True an array of edge weights is also returned.
            edge_types (list of hashable, optional): If provided, only traverse the graph
                via the provided edge types when collecting neighbours.
            use_ilocs (bool): if True `node` is treated as a :ref:`node iloc <iloc-explanation>`
                (and similarly `edge_types` is treated as a edge type ilocs) and the ilocs of each
                neighbour is returned.

        Returns:
            A numpy array of the neighboring out-nodes. If `include_edge_weight` is `True` then an array
            of edge weights is also returned in a tuple `(neighbor_array, edge_weight_array)`
        """
        if not self.is_directed():
            # all edges are both incoming and outgoing for undirected graphs
            return self.neighbor_arrays(
                node,
                include_edge_weight=include_edge_weight,
                edge_types=edge_types,
                use_ilocs=use_ilocs,
            )

        if not use_ilocs:
            node = self._nodes.ids.to_iloc([node])[0]

        edge_ilocs = self._edges.edge_ilocs(node, ins=False, outs=True)
        target = self._edges.targets[edge_ilocs]

        return self._transform_edges(
            target, edge_ilocs, include_edge_weight, edge_types, use_ilocs
        )

    def out_nodes(
        self, node: Any, include_edge_weight=False, edge_types=None, use_ilocs=False
    ) -> Iterable[Any]:
        """
        Obtains the collection of neighbouring nodes with edges
        directed from the given node. For an undirected graph,
        neighbours are treated as both in-nodes and out-nodes.

        Args:
            node (any): The node in question.
            include_edge_weight (bool, default False): If True, each neighbour in the
                output is a named tuple with fields `node` (the node ID) and `weight` (the edge weight)
            edge_types (list of hashable, optional): If provided, only traverse the graph
                via the provided edge types when collecting neighbours.
            use_ilocs (bool): if True `node` is treated as a :ref:`node iloc <iloc-explanation>`
                (and similarly `edge_types` is treated as a edge type ilocs) and the ilocs of each
                neighbour is returned.

        Returns:
            iterable: The neighbouring out-nodes.
        """
        neigh_arrs = self.out_node_arrays(
            node, include_edge_weight, edge_types, use_ilocs=use_ilocs
        )
        return self._to_neighbors(neigh_arrs, include_edge_weight)

    def nodes_of_type(self, node_type=None):
        """
        Get the nodes of the graph with the specified node types.

        Args:
            node_type (hashable): a type of nodes that exist in the graph (this must be passed,
                omitting it or passing ``None`` is deprecated)

        Returns:
            A list of node IDs with type node_type
        """
        warnings.warn(
            "'nodes_of_type' is deprecated and will be removed; use the 'nodes(type=...)' method instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return list(self.nodes(node_type=node_type))

    def node_type(self, node, use_ilocs=False):
        if is_real_iterable(node):
            nodes = node
        else:
            nodes = [node]

        if not use_ilocs:
            nodes = self._nodes.ids.to_iloc(nodes, strict=True)
        type_sequence = self._nodes.type_of_iloc(nodes)

        if is_real_iterable(node):
            return type_sequence

        assert len(type_sequence) == 1
        return type_sequence[0]

    @property
    def node_types(self):
        return set(self._nodes.types.pandas_index)

    def _unique_type(self, element_data, name, error_message):
        all_types = element_data.types.pandas_index
        if len(all_types) == 1:
            return all_types[0]

        found = comma_sep(all_types)
        if error_message is None:
            error_message = "Expected only one %(name)s type for 'unique_%(name)s_type', found: %(found)s"

        raise ValueError(error_message % {"name": name, "found": found})

    def unique_node_type(self, error_message=None):
        """
        Return the unique node type, for a homogeneous-node graph.

        Args:
            error_message (str, optional): a custom message to use for the exception; this can use
                the ``%(found)s`` placeholder to insert the real sequence of node types.

        Returns:
            If this graph has only one node type, this returns that node type, otherwise it raises a
            ``ValueError`` exception.
        """
        return self._unique_type(self._nodes, "node", error_message)

    @property
    def edge_types(self):
        return self._edges.types.pandas_index

    def unique_edge_type(self, error_message=None):
        return self._unique_type(self._edges, "edge", error_message)

    def node_type_names_to_ilocs(self, node_type_names):
        return self._nodes.types.to_iloc(node_type_names, strict=True)

    def node_type_ilocs_to_names(self, node_type_ilocs):
        return self._nodes.types.from_iloc(node_type_ilocs)

    def edge_type_names_to_ilocs(self, edge_type_names):
        return self._edges.types.to_iloc(edge_type_names, strict=True)

    def edge_type_ilocs_to_names(self, edge_type_ilocs):
        return self._edges.types.from_iloc(edge_type_ilocs)

    def _feature_shapes(self, element_data, types):
        all_sizes = element_data.feature_info()

        if types is None:
            types = all_sizes.keys()

        return {type_name: all_sizes[type_name][0] for type_name in types}

    def _feature_sizes(self, element_data, types, name):
        def get(type_name, shape):
            if len(shape) != 1:
                raise ValueError(
                    f"{name}_feature_sizes expects {name} types that have feature vectors (rank 1), found type {type_name!r} with feature shape {shape}"
                )

            return shape[0]

        return {
            type_name: get(type_name, shape)
            for type_name, shape in self._feature_shapes(element_data, types).items()
        }

    def node_feature_sizes(self, node_types=None):
        return self._feature_sizes(self._nodes, node_types, "node")

    def node_feature_shapes(self, node_types=None):
        return self._feature_shapes(self._nodes, node_types)

    def edge_feature_sizes(self, edge_types=None):
        return self._feature_sizes(self._edges, edge_types, "edge")

    def edge_feature_shapes(self, edge_types=None):
        return self._feature_shapes(self._edges, edge_types)

    def check_graph_for_ml(self, features=True, expensive_check=False):
        """
        Checks if all properties required for machine learning training/inference are set up.
        An error will be raised if the graph is not correctly setup.
        """
        if all(size == 0 for _, size in self.node_feature_sizes().items()):
            raise RuntimeError(
                "This StellarGraph has no numeric feature attributes for nodes"
                "Node features are required for machine learning"
            )

        # TODO: check the schema

        # TODO: check the feature node_ids against the graph node ids?

    def node_ids_to_ilocs(self, nodes):
        return self._nodes.ids.to_iloc(nodes, strict=True)

    def node_ilocs_to_ids(self, node_ilocs):
        return self._nodes.ids.from_iloc(node_ilocs)

    def node_features(self, nodes=None, node_type=None, use_ilocs=False):
        """
        Get the numeric feature vectors for the specified nodes or node type.

        For graphs with a single node type:

        - ``graph.node_features()`` to retrieve features of all nodes, in the same order as
          ``graph.nodes()``.

        - ``graph.node_features(nodes=some_node_ids)`` to retrieve features for each node in
          ``some_node_ids``.

        For graphs with multiple node types:

        - ``graph.node_features(node_type=some_type)`` to retrieve features of all nodes of type
          ``some_type``, in the same order as ``graph.nodes(node_type=some_type)``.

        - ``graph.node_features(nodes=some_node_ids, node_type=some_type)`` to retrieve features for
          each node in ``some_node_ids``. All of the chosen nodes must be of type ``some_type``.

        - ``graph.node_features(nodes=some_node_ids)`` to retrieve features for each node in
          ``some_node_ids``. All of the chosen nodes must be of the same type, which will be
          inferred. This will be slower than providing the node type explicitly in the previous example.

        Args:
            nodes (list or hashable, optional): Node ID or list of node IDs, all of the same type
            node_type (hashable, optional): the type of the nodes.

        Returns:
            Numpy array containing the node features for the requested nodes or node type.
        """
        return extract_element_features(
            self._nodes, self.unique_node_type, "node", nodes, node_type, use_ilocs
        )

    def edge_features(self, edges=None, edge_type=None, use_ilocs=False):
        """
        Get the numeric feature vectors for the specified edges or edge type.

        For graphs with a single edge type:

        - ``graph.edge_features()`` to retrieve features of all edges, in the same order as
          ``graph.edges()``.

        - ``graph.edge_features(edges=some_edge_ids)`` to retrieve features for each edge in
          ``some_edge_ids``.

        For graphs with multiple edge types:

        - ``graph.edge_features(edge_type=some_type)`` to retrieve features of all edges of type
          ``some_type``, in the same order as ``graph.edges(edge_type=some_type)``.

        - ``graph.edge_features(edges=some_edge_ids, edge_type=some_type)`` to retrieve features for
          each edge in ``some_edge_ids``. All of the chosen edges must be of type ``some_type``.

        - ``graph.edge_features(edges=some_edge_ids)`` to retrieve features for each edge in
          ``some_edge_ids``. All of the chosen edges must be of the same type, which will be
          inferred. This will be slower than providing the edge type explicitly in the previous example.

        Args:
            edges (list or hashable, optional): Edge ID or list of edge IDs, all of the same type
            edge_type (hashable, optional): the type of the edges.

        Returns:
            Numpy array containing the edge features for the requested edges or edge type.
        """
        return extract_element_features(
            self._edges, self.unique_edge_type, "edge", edges, edge_type, use_ilocs
        )

    ##################################################################
    # Computationally intensive methods:

    def _edge_type_iloc_triples(self, selector=slice(None), stacked=False):
        source_ilocs = self._edges.sources[selector]
        source_type_ilocs = self._nodes.type_ilocs[source_ilocs]

        rel_type_ilocs = self._edges.type_ilocs[selector]

        target_ilocs = self._edges.targets[selector]
        target_type_ilocs = self._nodes.type_ilocs[target_ilocs]

        all_ilocs = source_type_ilocs, rel_type_ilocs, target_type_ilocs
        if stacked:
            return np.stack(all_ilocs, axis=-1)

        return all_ilocs

    def _edge_type_triples(self, selector=slice(None)):
        src_ilocs, rel_ilocs, tgt_ilocs = self._edge_type_iloc_triples(
            selector, stacked=False
        )

        return (
            self._nodes.types.from_iloc(src_ilocs),
            self._edges.types.from_iloc(rel_ilocs),
            self._nodes.types.from_iloc(tgt_ilocs),
        )

    def _unique_type_triples(self, selector=slice(None)):
        all_type_ilocs = self._edge_type_iloc_triples(selector, stacked=True)

        if len(all_type_ilocs) == 0:
            # FIXME(https://github.com/numpy/numpy/issues/15559): if there's no edges, np.unique is
            # being called on a shape=(0, 3) ndarray, and hits "ValueError: cannot reshape array of
            # size 0 into shape (0,newaxis)", so we manually reproduce what would be returned
            ret = None, []
        else:
            ret = np.unique(all_type_ilocs, axis=0, return_index=True)

        edge_ilocs = ret[1]
        # we've now got the indices for an edge with each triple, along with the counts of them, so
        # we can query to get the actual edge types (this is, at the time of writing, easier than
        # getting the actual type for each type iloc in the triples)
        unique_ets = self._edge_type_triples(edge_ilocs)

        return zip(*unique_ets)

    def _edge_metrics_by_type_triple(self, metrics):
        src_ty, rel_ty, tgt_ty = self._edge_type_triples()
        df = pd.DataFrame(
            {
                "src_ty": src_ty,
                "rel_ty": rel_ty,
                "tgt_ty": tgt_ty,
                "weight": self._edges.weights,
            }
        )

        # if graph is undirected, we want to sort the triple
        if not self.is_directed():
            sorted_types = df[["src_ty", "tgt_ty"]].to_numpy()
            sorted_types.sort(axis=1)
            df[["src_ty", "tgt_ty"]] = sorted_types

        return df.groupby(["src_ty", "rel_ty", "tgt_ty"]).agg(metrics)["weight"]

    def info(self, show_attributes=None, sample=None, truncate=20):
        """
        Return an information string summarizing information on the current graph.
        This includes node and edge type information and their attributes.

        Note: This requires processing all nodes and edges and could take a long
        time for a large graph.

        Args:
            show_attributes: Deprecated, unused.
            sample: Deprecated, unused.
            truncate (int, optional): If an integer, show only the ``truncate`` most common node and
                edge type triples; if ``None``, list each one individually.
        Returns:
            An information string.
        """
        if show_attributes is not None:
            warnings.warn(
                "'show_attributes' is no longer used, remove it from the 'info()' call",
                DeprecationWarning,
                stacklevel=2,
            )

        if sample is not None:
            warnings.warn(
                "'sample' is no longer used, remove it from the 'info()' call",
                DeprecationWarning,
                stacklevel=2,
            )

        # always truncate the edge types listed for each node type, since they're redundant with the
        # individual listing of edge types, and make for a single very long line
        truncate_edge_types_per_node = 5
        if truncate is not None:
            truncate_edge_types_per_node = min(truncate_edge_types_per_node, truncate)

        # Numpy processing is much faster than NetworkX processing, so we don't bother sampling.
        gs = self.create_graph_schema()

        node_feature_info = self._nodes.feature_info()
        edge_feature_info = self._edges.feature_info()

        def str_edge_type(et):
            n1, rel, n2 = et
            return f"{n1}-{rel}->{n2}"

        def str_feature(feature_info, ty):
            feature_shape, feature_dtype = feature_info[ty]
            if len(feature_shape) > 1:
                return f"{feature_dtype.name} tensor, shape {feature_shape}"
            elif feature_shape[0] == 0:
                return "none"
            else:
                return f"{feature_dtype.name} vector, length {feature_shape[0]}"

        def str_node_type(count, nt):
            feature_text = str_feature(node_feature_info, nt)
            edges = gs.schema[nt]
            if edges:
                edge_types = comma_sep(
                    [str_edge_type(et) for et in gs.schema[nt]],
                    limit=truncate_edge_types_per_node,
                    stringify=str,
                )
            else:
                edge_types = "none"
            return f"{nt}: [{count}]\n    Features: {feature_text}\n    Edge types: {edge_types}"

        def edge_type_info(et, metrics):
            feature_text = str_feature(edge_feature_info, et.rel)
            if metrics.min == metrics.max:
                weights_text = (
                    "all 1 (default)" if metrics.min == 1 else f"all {metrics.min:.6g}"
                )
            else:
                weights_text = f"range=[{metrics.min:.6g}, {metrics.max:.6g}], mean={metrics.mean:.6g}, std={metrics.std:.6g}"

            return f"{str_edge_type(et)}: [{metrics.count}]\n        Weights: {weights_text}\n        Features: {feature_text}"

        # sort the node types in decreasing order of frequency
        node_types = sorted(
            ((len(self.nodes(node_type=nt)), nt) for nt in gs.node_types), reverse=True
        )
        nodes = separated(
            [str_node_type(count, nt) for count, nt in node_types],
            limit=truncate,
            stringify=str,
            sep="\n  ",
        )

        metric_names = ["count", "min", "max", "mean", "std"]
        et_metrics = self._edge_metrics_by_type_triple(metrics=metric_names)

        edge_types = sorted(
            (
                (metrics.count, EdgeType(*metrics.Index), metrics,)
                for metrics in et_metrics.itertuples()
            ),
            reverse=True,
        )

        edges = separated(
            [edge_type_info(et, metrics) for _, et, metrics in edge_types],
            limit=truncate,
            stringify=str,
            sep="\n    ",
        )

        directed_str = "Directed" if self.is_directed() else "Undirected"
        lines = [
            f"{type(self).__name__}: {directed_str} multigraph",
            f" Nodes: {self.number_of_nodes()}, Edges: {self.number_of_edges()}",
            "",
            " Node types:",
        ]
        if nodes:
            lines.append("  " + nodes)

        lines.append("")
        lines.append(" Edge types:")

        if edges:
            lines.append("    " + edges)

        return "\n".join(lines)

    def create_graph_schema(self, nodes=None):
        """
        Create graph schema from the current graph.

        Arguments:
            nodes (list): A list of node IDs to use to build schema. This must
                represent all node types and all edge types in the graph.
                If not specified, all nodes and edges in the graph are used.

        Returns:
            GraphSchema object.
        """

        graph_schema = {nt: set() for nt in self.node_types}
        edge_types = set()

        if nodes is None:
            selector = slice(None)
        else:
            node_ilocs = self._nodes.ids.to_iloc(nodes)
            selector = np.isin(self._edges.sources, node_ilocs) & np.isin(
                self._edges.targets, node_ilocs
            )

        for n1, rel, n2 in self._unique_type_triples(selector=selector):
            edge_type_tri = EdgeType(n1, rel, n2)
            edge_types.add(edge_type_tri)
            graph_schema[n1].add(edge_type_tri)

            if not self.is_directed():
                edge_type_tri = EdgeType(n2, rel, n1)
                edge_types.add(edge_type_tri)
                graph_schema[n2].add(edge_type_tri)

        # Create ordered list of edge_types
        edge_types = sorted(edge_types)

        # Create keys for node and edge types
        schema = {
            node_label: sorted(node_data)
            for node_label, node_data in graph_schema.items()
        }

        return GraphSchema(
            self.is_directed(), sorted(self.node_types), edge_types, schema
        )

    def node_degrees(self, use_ilocs=False) -> Mapping[Any, int]:
        """
        Obtains a map from node to node degree.

        use_ilocs (bool): if True return :ref:`node ilocs <iloc-explanation>`

        Returns:
            The degree of each node.
        """
        degrees = self._edges.degrees()
        if use_ilocs:
            return degrees
        node_ids = self.node_ilocs_to_ids(list(degrees.keys()))
        return defaultdict(
            int,
            ((node_id, degree) for node_id, degree in zip(node_ids, degrees.values())),
        )

    def to_adjacency_matrix(
        self, nodes: Optional[Iterable] = None, weighted=False, edge_type=None
    ):
        """
        Obtains a SciPy sparse adjacency matrix of edge weights.

        By default (``weighted=False``), each element of the matrix contains the number
        of edges between the two vertices (only 0 or 1 in a graph without multi-edges).

        Args:
            nodes (iterable): The optional collection of nodes
                comprising the subgraph. If specified, then the
                adjacency matrix is computed for the subgraph;
                otherwise, it is computed for the full graph.
            weighted (bool): If true, use the edge weight column from the graph instead
                of edge counts (weights from multi-edges are summed).
            edge_type (hashable, optional): If set (to an edge type), only includes edges of that
                type, otherwise uses all edges.

        Returns:
             The weighted adjacency matrix.
        """
        if edge_type is None:
            type_selector = slice(None)
        else:
            type_selector = self._edges.type_range(edge_type)

        sources = self._edges.sources[type_selector]
        targets = self._edges.targets[type_selector]

        if nodes is None:
            # if `nodes` is None use overall ilocs (for the original graph)
            src_idx = sources
            tgt_idx = targets
            selector = slice(None)
            n = self.number_of_nodes()
        else:
            node_ilocs = self._nodes.ids.to_iloc(nodes)
            index = ExternalIdIndex(node_ilocs)
            n = len(index)
            selector = np.isin(sources, node_ilocs) & np.isin(targets, node_ilocs)

            # these indices are computed relative to the index above
            src_idx = index.to_iloc(sources[selector])
            tgt_idx = index.to_iloc(targets[selector])

        if weighted:
            weights = self._edges.weights[type_selector][selector]
        else:
            weights = np.ones(src_idx.shape, dtype=self._edges.weights.dtype)

        adj = sps.csr_matrix((weights, (src_idx, tgt_idx)), shape=(n, n))
        if not self.is_directed() and n > 0:
            # in an undirected graph, the adjacency matrix should be symmetric: which means counting
            # weights from either "incoming" or "outgoing" edges, but not double-counting self loops

            # FIXME https://github.com/scipy/scipy/issues/11949: these operations, particularly the
            # diagonal, don't work for an empty matrix (n == 0)
            backward = adj.transpose(copy=True)
            # this is setdiag(0), but faster, since it doesn't change the sparsity structure of the
            # matrix (https://github.com/scipy/scipy/issues/11600)
            (nonzero,) = backward.diagonal().nonzero()
            backward[nonzero, nonzero] = 0

            adj += backward

        # this is a multigraph, let's eliminate any duplicate entries
        adj.sum_duplicates()
        return adj

    def subgraph(self, nodes):
        """
        Compute the node-induced subgraph implied by ``nodes``.

        Args:
            nodes (iterable): The nodes in the subgraph.

        Returns:
            A :class:`.StellarGraph` or :class:`.StellarDiGraph` instance containing only the nodes in
            ``nodes``, and any edges between them in ``self``. It contains the same node & edge
            types, node features and edge weights as in ``self``.
        """

        node_ilocs = self._nodes.ids.to_iloc(nodes, strict=True)
        node_types = self._nodes.type_of_iloc(node_ilocs)
        node_type_to_ilocs = pd.Series(node_ilocs, index=node_types).groupby(level=0)

        node_frames = {
            type_name: pd.DataFrame(
                self._nodes.features(type_name, ilocs),
                index=self._nodes.ids.from_iloc(ilocs),
            )
            for type_name, ilocs in node_type_to_ilocs
        }

        # FIXME(#985): this is O(edges in graph) but could potentially be optimised to O(edges in
        # graph incident to `nodes`), which could be much fewer if `nodes` is small
        edge_ilocs = np.where(
            np.isin(self._edges.sources, node_ilocs)
            & np.isin(self._edges.targets, node_ilocs)
        )
        edge_frame = pd.DataFrame(
            {
                "id": self._edges.ids.from_iloc(edge_ilocs),
                globalvar.SOURCE: self._nodes.ids.from_iloc(
                    self._edges.sources[edge_ilocs]
                ),
                globalvar.TARGET: self._nodes.ids.from_iloc(
                    self._edges.targets[edge_ilocs]
                ),
                globalvar.WEIGHT: self._edges.weights[edge_ilocs],
            },
            index=self._edges.type_of_iloc(edge_ilocs),
        )
        edge_frames = {
            type_name: df.set_index("id")
            for type_name, df in edge_frame.groupby(level=0)
        }

        if self.is_directed():
            raise ValueError("Directed Graph is not supported")
        else:
            cls = StellarGraph
        return cls(node_frames, edge_frames)

    def connected_components(self):
        """
        Compute the connected components in this graph, ordered by size.

        The nodes in the largest component can be computed with ``nodes =
        next(graph.connected_components())``. The node IDs returned by this method can be used to
        compute the corresponding subgraph with ``graph.subgraph(nodes)``.

        For directed graphs, this computes the weakly connected components. This effectively
        treating each edge as undirected.

        Returns:
            An iterator over sets of node IDs in each connected component, from the largest (most nodes)
            to smallest (fewest nodes).
        """

        adj = self.to_adjacency_matrix()
        count, cc_labels = sps.csgraph.connected_components(adj, directed=False)
        cc_sizes = np.bincount(cc_labels, minlength=count)
        cc_by_size = np.argsort(cc_sizes)[::-1]

        return (
            self._nodes.ids.from_iloc(cc_labels == cc_label) for cc_label in cc_by_size
        )

    def _adjacency_types(self, graph_schema: GraphSchema, use_ilocs=False):
        """
        Obtains the edges in the form of the typed mapping:

            {edge_type_triple: {source_node: [target_node, ...]}}

        Args:
            graph_schema: The graph schema.
            use_ilocs (bool): if True return :ref:`node ilocs <iloc-explanation>`
        Returns:
             The edge types mapping.
        """
        source_types, rel_types, target_types = self._edge_type_triples(slice(None))

        triples = defaultdict(lambda: defaultdict(lambda: []))

        sources = self._edges.sources
        targets = self._edges.targets
        if not use_ilocs:
            sources = self._nodes.ids.from_iloc(sources)
            targets = self._nodes.ids.from_iloc(targets)

        iterator = zip(source_types, rel_types, target_types, sources, targets,)
        for src_type, rel_type, tgt_type, src, tgt in iterator:
            triple = EdgeType(src_type, rel_type, tgt_type)
            triples[triple][src].append(tgt)

            if not self.is_directed() and src != tgt:
                other_triple = EdgeType(tgt_type, rel_type, src_type)
                triples[other_triple][tgt].append(src)

        for subdict in triples.values():
            for v in subdict.values():
                # each list should be in order, to ensure sampling methods are deterministic
                v.sort(key=str)

        return triples

    def _edge_weights(
        self, source_node: Any, target_node: Any, use_ilocs=False
    ) -> List[Any]:
        """
        Obtains the weights of edges between the given pair of nodes.

        Args:
            source_node (int): The source node.
            target_node (int): The target node.
            use_ilocs (bool): if True source_node and target_node are treated as :ref:`node ilocs <iloc-explanation>`.

        Returns:
            list: The edge weights.
        """
        # self loops should only be counted once, which means they're effectively always a directed
        # edge at the storage level, unlikely other edges in an undirected graph. This is
        # particularly important with the intersection1d call, where the source_ilocs and
        # target_ilocs will be equal, when source_node == target_node, and thus the intersection
        # will contain all incident edges.
        effectively_directed = self.is_directed() or source_node == target_node
        both_dirs = not effectively_directed

        if not use_ilocs:
            source_node = self._nodes.ids.to_iloc([source_node])[0]
            target_node = self._nodes.ids.to_iloc([target_node])[0]

        source_edge_ilocs = self._edges.edge_ilocs(
            source_node, ins=both_dirs, outs=True
        )
        target_edge_ilocs = self._edges.edge_ilocs(
            target_node, ins=True, outs=both_dirs
        )

        ilocs = np.intersect1d(source_edge_ilocs, target_edge_ilocs, assume_unique=True)

        return [float(x) for x in self._edges.weights[ilocs]]


