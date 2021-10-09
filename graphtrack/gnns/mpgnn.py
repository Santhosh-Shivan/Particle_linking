import tensorflow as tf
from tensorflow.keras import layers

from ..deeptrack.models.utils import KerasModel
from .layers import as_block

from .layers import GraphLayer as graphblock


class mpGraphNet(KerasModel):
    """
    Message passing graph neural network.
    Parameters:
    -----------
    dense_layer_dimensions: list of ints
        List of the number of nodes in each dense layer.
    base_layer_dimensions: list of ints
        List of the number of nodes in each base layer.
    number_of_node_outputs: int
        Number of output node features.
    number_of_edge_outputs: int
        Number of output edge features.
    node_output_activation: str
        Activation function for the output node layer.
    edge_output_activation: str
        Activation function for the output edge layer.
    dense_block: str
        Name of the dense block.
    output_type: str
        Type of output. Either "nodes", "edges", or "graph".
        If 'key' is not a supported output type, then the
        model output will be the concatenation of the node
        and edge predictions.
    graph_block: str
        Name of the graph block.
    kwargs: dict
        Keyword arguments for the dense block.
    Returns:
    --------
    tf.keras.Model
        Keras model for the graph neural network.
    """

    def __init__(
        self,
        dense_layer_dimensions=(32, 72),
        base_layer_dimensions=(72, 72),
        number_of_node_features=7,
        number_of_edge_features=7,
        number_of_node_outputs=1,
        number_of_edge_outputs=1,
        node_output_activation=None,
        edge_output_activation=None,
        dense_block="graphdense",
        output_type="graph",
        loss=[
            tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        ]
        * 2,
        **kwargs
    ):

        dense_block = as_block(dense_block)
        graph_block = graphblock

        node_features, edge_features, edges, edge_weights = (
            tf.keras.Input(shape=(None, number_of_node_features)),
            tf.keras.Input(shape=(None, number_of_edge_features)),
            tf.keras.Input(shape=(None, 2), dtype=tf.int32),
            tf.keras.Input(shape=(None, 2)),
        )

        node_layer = node_features
        edge_layer = edge_features

        # Create seperate graph encoder for node and edge features
        for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layer_dimensions)), dense_layer_dimensions
        ):
            node_layer = dense_block(
                dense_layer_dimension,
                name="node_ide" + str(dense_layer_number + 1),
                **kwargs
            )(node_layer)

            edge_layer = dense_block(
                dense_layer_dimension,
                name="edge_ide" + str(dense_layer_number + 1),
                **kwargs
            )(edge_layer)

        distance = edge_features[..., 0]
        layer = (node_layer, edge_layer, distance, edges, edge_weights)

        for base_layer_number, base_layer_dimension in zip(
            range(len(base_layer_dimensions)), base_layer_dimensions
        ):
            layer = graph_block(
                base_layer_dimension,
                name="graph_block_" + str(base_layer_number),
            )(layer)

        node_layer, edge_layer, *_ = layer

        for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layer_dimensions)),
            reversed(dense_layer_dimensions),
        ):
            node_layer = dense_block(
                dense_layer_dimension,
                name="node_idd" + str(dense_layer_number + 1),
                **kwargs
            )(node_layer)

            edge_layer = dense_block(
                dense_layer_dimension,
                name="edge_idd" + str(dense_layer_number + 1),
                **kwargs
            )(edge_layer)

        # Output layers
        node_output = layers.Dense(
            number_of_node_outputs,
            activation=node_output_activation,
            name="node_prediction",
        )(node_layer)

        edge_output = layers.Dense(
            number_of_edge_outputs,
            activation=edge_output_activation,
            name="edge_prediction",
        )(edge_layer)

        output_dict = {
            "nodes": node_output,
            "edges": edge_output,
            "graph": [node_output, edge_output],
        }
        try:
            outputs = output_dict[output_type]
        except KeyError:
            outputs = output_dict["graph"]

        model = tf.keras.models.Model(
            [node_features, edge_features, edges, edge_weights],
            outputs,
        )

        super().__init__(model, loss=loss, **kwargs)
