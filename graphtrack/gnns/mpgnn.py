import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.ops.variables import trainable_variables

from ..deeptrack.models.utils import as_KerasModel
from .layers import as_block


@as_KerasModel
class mpGraphNet(tf.keras.Model):
    """
    Message passing graph neural network.
    Parameters:
    -----------
    dense_layer_dimensions: list of ints
        List of the number of nodes in each dense layer.
    base_layer_dimensions: list of ints
        List of the number of nodes in each base layer.
    number_of_outputs: int
        Number of output nodes.
    output_activation: str
        Activation function for the output layer.
    dense_block: str
        Name of the dense block.
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
        number_of_outputs=3,
        output_activation=None,
        dense_block="_dense",
        graph_block="mpgblock",
        **kwargs
    ):
        super().__init__()

        dense_block = as_block(dense_block)
        graph_block = as_block(graph_block)

        # Create seperate graph encoder for node and edge features
        node_encoder, edge_encoder = [], []
        for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layer_dimensions)), dense_layer_dimensions
        ):
            node_encoder.append(
                dense_block(
                    dense_layer_dimension,
                    name="node_ide" + str(dense_layer_number + 1),
                    **kwargs
                )
            )
            edge_encoder.append(
                dense_block(
                    dense_layer_dimension,
                    name="edge_ide" + str(dense_layer_number + 1),
                    **kwargs
                )
            )
        self.node_encoder = tf.keras.Sequential(node_encoder)
        self.edge_encoder = tf.keras.Sequential(edge_encoder)

        # Message passing graph neural network as base layer
        self.graph_layers = []
        for base_layer_number, base_layer_dimension in zip(
            range(len(base_layer_dimensions)), base_layer_dimensions
        ):
            self.graph_layers.append(
                graph_block(
                    base_layer_dimension,
                    name="graph_block_" + str(base_layer_number),
                )
            )

        # Create seperate graph decoder for node and edge features
        node_decoder, edge_decoder = [], []
        for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layer_dimensions)),
            reversed(dense_layer_dimensions),
        ):
            node_decoder.append(
                dense_block(
                    dense_layer_dimension,
                    name="node_idd" + str(dense_layer_number + 1),
                    **kwargs
                )
            )
            edge_decoder.append(
                dense_block(
                    dense_layer_dimension,
                    name="edge_idd" + str(dense_layer_number + 1),
                    **kwargs
                )
            )
        self.node_decoder = tf.keras.Sequential(node_decoder)
        self.edge_decoder = tf.keras.Sequential(edge_decoder)

        # Output layers
        self.nodes_output_layer = layers.Dense(
            number_of_outputs, activation=output_activation
        )
        self.edges_output_layer = layers.Dense(
            number_of_outputs, activation=output_activation
        )

    # def train_step(self, data):
    #     batch, labels = data

    #     with tf.GradientTape() as tape:
    #         pred = self(batch, training=True)

    #         # Compute loss
    #         loss = self.compiled_loss(
    #             labels,
    #             pred,
    #         )

    #         # Compute gradients
    #         trainable_vars = self.trainable_variables
    #         grads = tape.gradient(loss, trainable_vars)
    #         # Update weights
    #         self.optimizer.apply_gradients(zip(grads, trainable_vars))

    #         loss = {"loss": loss}

    #         return loss

    def call(self, inputs):
        """
        Forward pass of the graph neural network.
        Parameters:
        -----------
        inputs: List of tensors
            Node features, edge features and graph adjacency matrix.
        Returns:
        --------
        List of tensors
            Node and edge features.
        """

        nodes, edge_features, edges, edge_weights = inputs

        print(nodes.shape)

        # Encode node and edge features
        encoded_nodes = self.node_encoder(nodes)
        encoded_edge_features = self.edge_encoder(edge_features)

        # Add new axis to the node features, edge features
        # and graph adjacency matrix before passing to the
        # graph neural network
        layer = [
            encoded_nodes,
            encoded_edge_features,
            edges,
            edge_weights if not (edge_weights is None) else None,
        ]

        # Apply graph neural network
        for graph_layer in self.graph_layers:
            layer = graph_layer(layer)

        # Squeeze the output of the graph neural network
        nodes_latent, edge_features_latent, *_ = map(
            lambda x: tf.squeeze(x) if not (x is None) else None, layer
        )

        # Decode node and edge features
        decoded_nodes = self.node_decoder(nodes_latent)
        decoded_edges = self.node_decoder(edge_features_latent)

        # Return the output of the graph neural network
        outputs = (
            self.nodes_output_layer(decoded_nodes),
            self.edges_output_layer(decoded_edges),
        )
        return outputs
