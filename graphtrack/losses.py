import tensorflow as tf


def GraphCategoricalCrossEntropy(
    from_logits=False, loss_weights=[1, 1], **kwargs
):
    """
    The categorical cross entropy loss function for graphs.
    ----------
    from_logits : bool
        If True, the input node and edge features are
        assumed to be the logits of the network, otherwise
        it is assumed to be the probabilities.
    loss_weights : list of floats
        The weights of the loss function for nodes and edges.
    kwargs : dict
        The keyword arguments to pass to the tensorflow
        function.
    """
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=from_logits, **kwargs
    )

    def loss(T, P):
        """
        Computes the categorical cross entropy loss
        over nodes and edges in a graph.
        Parameters
        ----------
        T : tuple of tensors
            The first element of the tuple is the
            tensor of node features, and the second is
            the tensor of edge features. The tensors
            are of shape (batch_size, num_nodes,
            num_features).
        P : tuple of tensors
            The first element of the tuple is the
            tensor of node labels, and the second is
            the tensor of edge labels. The tensors
            are of shape (batch_size, num_nodes,
            num_classes).
        """
        # Compute the loss over the nodes.
        node_loss = loss_fn(T[0], P[0])

        # Compute the loss over the edges.
        edge_loss = loss_fn(T[1], P[1])

        # Return the sum of the losses.
        loss = loss_weights[0] * node_loss + loss_weights[1] * edge_loss
        return loss

    return loss
