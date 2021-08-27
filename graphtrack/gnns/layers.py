""" Standardized layers implemented in keras for GNNs.
"""


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


class KerasLayerWrapper(tf.keras.layers.Layer):
    """
    Keras layer wrapper for a single layer.
    """

    def __init__(self, layer, layer_norm, activation, **kwargs):
        super().__init__(**kwargs)
        self.layer, self.layer_norm, self.activation = (
            layer,
            layer_norm,
            activation,
        )

    def call(self, inputs):
        y = self.layer(inputs)

        if self.activation:
            y = _as_activation(self.activation)(y)

        if self.layer_norm:
            if not isinstance(self.layer_norm, dict):
                layer_norm = {}
            y = layers.LayerNormalization(**layer_norm)(y)

        return y


def _layer_norm(x, filters):
    if callable(x):
        return x(filters)
    else:
        return x


@register("_dense")
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
        return KerasLayerWrapper(
            layer,
            _layer_norm(layer_norm, filters),
            activation,
            **kwargs_inner,
        )

    return Layer


@register("grublock")
def gruBlock(
    activation=None,
    layer_norm=True,
    dropout=0.2,
    return_sequences=True,
    **kwargs,
):
    """A single gru layer.

    Accepts arguments of keras.layers.GRU.

    Parameters
    ----------
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    layer_norm : bool
        Whether to add layer normalization (after activation).
    **kwargs
        Other keras.layers.Dense arguments
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)

        layer = layers.GRU(
            units=filters,
            dropout=dropout,
            return_sequences=return_sequences,
            **kwargs_inner,
        )

        return lambda x: _single_layer_call(
            x, layer, _layer_norm(layer_norm, filters), activation
        )

    return Layer


@register("classtoken")
def ClassToken(*args, **kwargs):
    """Append a class token to an input layer."""

    def Layer(*args, **kwargs_inner):
        kwargs_inner.update(kwargs)

        cls_init = tf.zeros_initializer()

        def call(x):
            batch_size = x.shape[0]
            filters = x.shape[-1]

            _cls = tf.Variable(
                name="cls",
                initial_value=cls_init(shape=(1, 1, filters), dtype="float32"),
                trainable=True,
            )

            cls_broadcasted = tf.cast(
                tf.broadcast_to(_cls, [batch_size, 1, filters]),
                dtype=x.dtype,
            )
            return tf.concat([cls_broadcasted, x], 1)

        return call

    return Layer


@register("multiheadatt")
def MultiHeadSelfAttention(num_heads=12, **kwargs):
    """Multi-head self-attention layer.
    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    kwargs
        Other keras.layers.Dense arguments
    """

    def SingleAttention(query, key, value):
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
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(x, batch_size, projection_dim):
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
        x = tf.reshape(x, (batch_size, -1, num_heads, projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def Layer(**kwargs_inner):
        kwargs_inner.update(kwargs)

        def call(x):
            nodes, aggregated = x

            batch_size = nodes.shape[0]
            filters = aggregated.shape[-1]

            if filters % num_heads != 0:
                raise ValueError(
                    f"embedding dimension = {filters} should be divisible by number of heads = {num_heads}"
                )

            projection_dim = filters // num_heads

            x = tf.concat([nodes, aggregated], axis=-1)

            query = layers.Dense(filters)(x)
            key = layers.Dense(filters)(x)
            value = layers.Dense(filters)(x)

            query = separate_heads(query, batch_size, projection_dim)
            key = separate_heads(key, batch_size, projection_dim)
            value = separate_heads(value, batch_size, projection_dim)

            attention, weights = SingleAttention(query, key, value)
            attention = tf.transpose(attention, perm=[0, 2, 1, 3])
            concat_attention = tf.reshape(attention, (batch_size, -1, filters))
            output = layers.Dense(filters)(concat_attention)

            return output

        return call

    return Layer


class KerasGNNLayerWrapper(tf.keras.layers.Layer):
    """
    Keras layer wrapper for Graph Neural Network layer.
    Parameters
    ----------
    layer : tf.keras.layers.Layer
        Keras layer.
    filters : int
        Number of output filters.
    message_layer : tf.keras.layers.Layer
        Message passing layer.
    update_layer : tf.keras.layers.Layer
        Update layer.
    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    layer_norm : bool
        Whether to add layer normalization (after activation).
    random_edge_dropout : bool
        Whether to drop edges randomly.
    kwargs
        Other keras.layers.Dense arguments
    Returns
    -------
    keras.layers.Layer
        Keras GNN layer.
    """

    def __init__(
        self,
        filters,
        message_layer,
        update_layer,
        activation,
        layer_norm,
        random_edge_dropout,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.message_layer = message_layer
        self.update_layer = update_layer
        self.activation = activation
        self.layer_norm = layer_norm
        self.random_edge_dropout = random_edge_dropout

    def call(self, inputs):
        nodes, edge_features, edges, edge_weights = inputs
        print(nodes.shape)
        # nodes, edge_features, edges, edge_weights = (
        #     nodes[0, ...],
        #     edge_features[0, ...],
        #     edges[0, ...],
        #     edge_weights[0, ...] if not (edge_weights is None) else None,
        # )

        nOfnodes, nOffeatures, nOfedges, batch_size = (
            tf.shape(nodes)[1],
            tf.shape(nodes)[-1],
            tf.shape(edges)[1],
            tf.shape(nodes)[0],
        )

        # Get neighbors node features, shape = (batch, nOfedges, 2, nOffeatures)
        message_inputs = tf.gather(nodes, edges, batch_dims=1)

        # Concatenate nodes features with edge features,
        # shape = (batch, nOfedges, 2*nOffeatures + nOffedgefeatures)
        messages = tf.reshape(
            message_inputs, (batch_size, nOfedges, 2 * nOffeatures)
        )
        reshaped = tf.concat(
            [
                messages,
                edge_features,
            ],
            -1,
        )

        # Compute messages/update edges, shape = (batch, nOfedges, filters)
        messages = _single_layer_call(
            reshaped,
            self.message_layer,
            _layer_norm(self.layer_norm, self.filters),
            self.activation,
        )
        if self.random_edge_dropout:
            messages = tf.nn.dropout(
                messages,
                self.random_edge_dropout,
                noise_shape=(1, nOfedges, 1),
            )

        if not (edge_weights is None):

            # If weights are provided, apply them to the messages
            # shape = (batch, nOfedges, filters)
            weighted_messages = messages * edge_weights[..., 1:2]

            # Merge repeated edges, shape = (batch, nOfedges (before augmentation), filters)
            weighted_messages = [
                tf.math.unsorted_segment_sum(
                    weighted_messages[idx],
                    tf.cast(edge_weights[idx, ..., 0], tf.int32),
                    num_segments=tf.shape(
                        tf.unique(edge_weights[idx, ..., 0])[0]
                    )[0],
                )
                for idx in range(batch_size)
            ]

            # Aggregate messages, shape = (batch, nOfnode, filters)
            aggregated = [
                tf.math.unsorted_segment_sum(
                    weighted_messages[idx],
                    edges[idx, : tf.shape(weighted_messages[idx])[0], 1],
                    nOfnodes,
                )
                for idx in range(batch_size)
            ]
        else:
            # If no weights are provided, aggregate messages
            # shape = (batch, nOfnode, filters)
            aggregated = [
                tf.math.unsorted_segment_sum(
                    messages[idx], edges[idx, ..., 1], nOfnodes
                )
                for idx in range(batch_size)
            ]

        aggregated = tf.stack(aggregated, axis=0)

        # Update node features, (nOfnode, filters)
        Combined = [nodes, aggregated]
        updated_nodes = _single_layer_call(
            Combined,
            self.update_layer,
            _layer_norm(self.layer_norm, self.filters),
            self.activation,
        )
        return (
            updated_nodes,
            messages,
            edges,
            edge_weights if not (edge_weights is None) else None,
        )


@register("mpgblock")
def mpGraphBlock(
    activation=tfa.activations.gelu,
    layer_norm=True,
    update_type="Transformer",
    random_edge_dropout=None,
    **kwargs,
):
    """A single message passing graph layer

    Can optionally perform some activation function. Accepts arguments
    of keras.layers.Layer.

    Parameters
    ----------

    activation : str or activation function or layer
        Activation function of the layer. See keras docs for accepted strings.
    **kwargs
        Other keras.layers.Layer arguments
    """

    def Layer(filters, **kwargs_inner):
        kwargs_inner.update(kwargs)

        # Message function
        message_layer = layers.Dense(filters, **kwargs_inner)

        # Update function
        if update_type == "GRU":
            update_layer = layers.GRU(
                units=filters, return_sequences=True, **kwargs_inner
            )
        elif update_type == "Transformer":
            update_layer = MultiHeadSelfAttention(
                num_heads=kwargs_inner.get("num_heads", 12),
            )()
        else:
            raise ValueError(
                f"Invalid update type: {update_type}. update_type must be either GRU or Transformer"
            )

        return KerasGNNLayerWrapper(
            filters,
            message_layer,
            update_layer,
            activation,
            layer_norm,
            random_edge_dropout,
            **kwargs_inner,
        )

    return Layer
