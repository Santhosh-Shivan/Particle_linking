import tensorflow as tf


class GraphAccuracy(tf.keras.metrics.Accuracy):
    """
    The accuracy metric for graphs.
    """

    def __init__(self, index=0, name=None, **kwargs):
        super(GraphAccuracy, self).__init__(name=name, **kwargs)

        #  The index of the node/edge tensor. If index is 0,
        #  then the accuracy will be computed over the first
        #  element of the tuple, i.e, the nodes. If index is 1,
        #  then the accuracy will be computed over the second
        #  element of the tuple, i.e, the edges.
        self.index = index

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Computes the accuracy metric over
        nodes/edges in a graph.
        Parameters
        ----------
        y_true : tuple of tensors
            The first element of the tuple is the
            tensor of node features, and the second is
            the tensor of edge features. The tensors
            are of shape (batch_size, num_nodes,
            num_features).
        y_pred : tuple of tensors
            The first element of the tuple is the
            tensor of node labels, and the second is
            the tensor of edge labels. The tensors
            are of shape (batch_size, num_nodes,
            num_classes).
        """
        y_true = tf.argmax(y_true[self.index], axis=-1)
        y_pred = tf.argmax(y_pred[self.index], axis=-1)

        # Update metric
        return super(GraphAccuracy, self).update_state(y_true, y_pred)
