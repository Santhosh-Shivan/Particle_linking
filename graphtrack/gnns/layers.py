""" Standardized layers implemented in keras for GNNs.
"""


from logging import lastResort
from warnings import WarningMessage
from tensorflow.keras import activations, layers
import tensorflow as tf
from tensorflow.python.keras.backend import dtype
import tensorflow_addons as tfa
from tensorflow.python.keras.layers import normalization
from tensorflow.python.ops.nn_impl import normalize

try:
    from tensorflow_addons.layers import InstanceNormalization
except Exception:
    import warnings

    InstanceNormalization = layers.Layer
    warnings.warn(
        "DeepTrack not installed with tensorflow addons. Instance normalization will not work. Consider upgrading to tensorflow >= 2.0.",
        ImportWarning,
    )


BLOCKS = {}


def register(*names):
    "Register a block to a name for use in models."

    def decorator(block):
        for name in names:
            if name in BLOCKS:
                warnings.warn(
                    f"Overriding registered block {name} with a new block.",
                    WarningMessage,
                )

            BLOCKS[name] = block()
        return block

    return decorator


def as_block(x):
    """Converts input to layer block"""
    if isinstance(x, str):
        if x in BLOCKS:
            return BLOCKS[x]
        else:
            raise ValueError(
                "Invalid blockname {0}, valid names are: ".format(x)
                + ", ".join(BLOCKS.keys())
            )
    if isinstance(x, layers.Layer) or not callable(x):
        raise TypeError(
            "Layer block should be a function that returns a keras Layer."
        )
    else:
        return x


def _as_activation(x):
    if x is None:
        return layers.Layer()
    elif isinstance(x, str):
        return layers.Activation(x)
    elif isinstance(x, layers.Layer):
        return x
    else:
        return layers.Layer(x)


def _single_layer_call(x, layer, layer_norm, activation):
    y = layer(x)

    if activation:
        y = _as_activation(activation)(y)

    if layer_norm:
        if not isinstance(layer_norm, dict):
            layer_norm = {}
        y = layers.LayerNormalization(**layer_norm)(y)

    return y


def _layer_norm(x, filters):
    if callable(x):
        return x(filters)
    else:
        return x


@register("graphdense")
def DenseBlock_(activation=tfa.activations.gelu, layer_norm=False, **kwargs):
    """A single dense layer.
    Accepts arguments of keras.layers.Dense.
    Parameters
    ----------
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    instance_norm : bool
        Whether to add instance normalization (before activation).
    **kwargs
        Other keras.layers.Dense arguments
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)
        layer = layers.Dense(filters, **kwargs_inner)
        return lambda x: _single_layer_call(
            x,
            layer,
            _layer_norm(layer_norm, filters),
            activation,
        )

    return Layer


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """Multi-head self-attention layer.
    Parameters
    ----------
    number_of_heads : int
        Number of attention heads.
    kwargs
            Other keras.layers.Dense arguments
    """

    def __init__(self, number_of_heads, **kwargs):
        super().__init__(**kwargs)
        self.number_of_heads = number_of_heads

    def build(self, input_shape):
        """
        Build the layer.
        """
        try:
            filters = input_shape[1][-1]
        except TypeError:
            filters = input_shape[-1]

        if filters % self.number_of_heads != 0:
            raise ValueError(
                f"embedding dimension = {filters} should be divisible by number of heads = {self.number_of_heads}"
            )
        self.filters = filters
        self.projection_dim = filters // self.number_of_heads

        self.query_dense = layers.Dense(filters)
        self.key_dense = layers.Dense(filters)
        self.value_dense = layers.Dense(filters)
        self.combine_dense = layers.Dense(filters)

        self.att_weights = tf.Variable(
            tf.zeros((1, self.number_of_heads, 1, 1)),
            name="attention_weights",
            dtype=tf.float32,
            shape=tf.TensorShape([None, self.number_of_heads, None, None]),
        )

    def SingleAttention(self, query, key, value, softmax_norm=True, **kwargs):
        """
        Single attention layer.
        Parameters
        ----------
        query : tf.Tensor
            Query tensor.
        key : tf.Tensor
            Key tensor.
        value : tf.Tensor
            Value tensor.
        """
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)

        if not softmax_norm:
            return scaled_score

        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        """
        Separate heads.
        Parameters
        ----------
        x : tf.Tensor
            Input tensor.
        batch_size : int
            Batch size.
        projection_dim : int
            Projection dimension.
        """
        x = tf.reshape(
            x, (batch_size, -1, self.number_of_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def compute_attention(self, x, **kwargs):
        """
        Compute attention.
        Parameters
        ----------
        x : tf.Tensor
            Input tensor.
        kwargs
            Other arguments to pass to SingleAttention.
        """
        if not isinstance(x, list):
            x = [x]

        x = tf.concat(x, axis=-1)
        batch_size = tf.shape(x)[0]

        query = self.query_dense(x)
        key = self.key_dense(x)
        value = self.value_dense(x)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        return (self.SingleAttention(query, key, value, **kwargs), batch_size)

    def call(self, x, **kwargs):
        """
        Call the layer.
        Parameters
        ----------
        x : tuple of tf.Tensors
            Input tensors.
        """
        (attention, att_weights), batch_size = self.compute_attention(
            x, **kwargs
        )
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.filters)
        )
        output = self.combine_dense(concat_attention)
        self.att_weights.assign(att_weights)

        return output


class GraphLayer(tf.keras.layers.Layer):
    """
    Message passing layer.
    Parameters
    ----------
    filters : int
        Number of filters.
    number_of_heads : int
        Number of attention heads.
    random_edge_dropout : float, optional
        Random edge dropout.
    """

    def __init__(
        self,
        filters,
        number_of_heads=12,
        random_edge_dropout=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.number_of_heads = number_of_heads
        self.random_edge_dropout = random_edge_dropout

        self.message_layer = tf.keras.Sequential(
            [
                layers.Dense(self.filters),
                tf.keras.layers.Lambda(
                    lambda x: tfa.activations.gelu(x, approximate=False)
                ),
                layers.LayerNormalization(),
            ]
        )

        self.update_layer = tf.keras.Sequential(
            [
                MultiHeadSelfAttention(
                    number_of_heads=self.number_of_heads,
                ),
                tf.keras.layers.Lambda(
                    lambda x: tfa.activations.gelu(x, approximate=False)
                ),
                layers.LayerNormalization(),
            ]
        )

    def build(self, input_shape):
        self.sigma = tf.Variable(
            initial_value=tf.constant_initializer(value=0.01)(
                shape=(1,), dtype="float32"
            ),
            name="sigma",
            trainable=True,
            constraint=lambda value: tf.clip_by_value(value, 0.002, 0.1),
        )

        self.beta = tf.Variable(
            initial_value=tf.constant_initializer(value=1)(
                shape=(1,), dtype="float32"
            ),
            name="beta",
            trainable=True,
            constraint=lambda value: tf.clip_by_value(value, 1, 4),
        )

    def call(self, inputs):
        nodes, edge_features, distance, edges, _ = inputs

        number_of_nodes = tf.shape(nodes)[1]
        number_of_edges = tf.shape(edges)[1]
        number_of_node_features = nodes.shape[-1]

        batch_size = tf.shape(nodes)[0]

        # Get neighbors node features, shape = (batch, nOfedges, 2, nOffeatures)
        message_inputs = tf.gather(nodes, edges, batch_dims=1)

        # Concatenate nodes features with edge features,
        # shape = (batch, nOfedges, 2*nOffeatures + nOffedgefeatures)
        messages = tf.reshape(
            message_inputs,
            (
                batch_size,
                number_of_edges,
                2 * number_of_node_features,
            ),
        )
        reshaped = tf.concat(
            [
                messages,
                edge_features,
            ],
            -1,
        )

        # Compute messages/update edges, shape = (batch, nOfedges, filters)
        messages = self.message_layer(reshaped)

        if self.random_edge_dropout:
            messages = tf.nn.dropout(
                messages,
                self.random_edge_dropout,
                noise_shape=(1, number_of_edges, 1),
            )

        # Compute edge weights and apply them to the messages
        # shape = (batch, nOfedges, filters)
        edge_weights = tf.math.exp(
            -1 * ((distance ** 2) / (2 * self.sigma ** 2)) ** self.beta
        )
        weighted_messages = messages * edge_weights[..., tf.newaxis]

        # Merge repeated edges, shape = (batch, nOfedges (before augmentation), filters)
        def aggregate(_, x):
            message, edge = x

            merged_edges = tf.math.unsorted_segment_sum(
                message,
                edge[: tf.shape(message)[0], 1],
                number_of_nodes,
            )

            return merged_edges

        # Aggregate messages, shape = (batch, nOfnodes, filters)
        aggregated = tf.scan(
            aggregate,
            (weighted_messages, edges),
            initializer=tf.zeros((number_of_nodes, number_of_node_features)),
        )

        # Update node features, (nOfnode, filters)
        Combined = [nodes, aggregated]
        updated_nodes = self.update_layer(Combined)

        return (
            updated_nodes,
            weighted_messages,
            distance,
            edges,
            edge_weights,
        )
