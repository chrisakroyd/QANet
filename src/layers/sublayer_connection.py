import tensorflow as tf
from tensorflow.keras.layers import Dropout, Layer


class SublayerConnection(Layer):
    def __init__(self, dropout, sublayer, total_sublayers, **kwargs):
        """ Layer Dropout implementation
            Adds a Layer Dropout layer, based on the paper "Deep Networks with Stochastic Depth"
            (https://arxiv.org/abs/1603.09382). Functions by dropping out entire layers with
            probability P that increases for layers deeper within the network.

            This layer is also responsible for the residual connection between layers, when
            P is great enough and we opt to dropout the layer, we simply return the
            residual.

            @TODO Ensure that sublayer + total_sublayers generation isn't skipping values.

            Args:
                dropout: P of dropping a layer.
                sublayer: Integer or float value representing this layers position.
                total_sublayers: The total number of layers.
        """
        super(SublayerConnection, self).__init__(**kwargs)
        self.Pl = dropout * float(sublayer) / float(total_sublayers)
        self.dropout = Dropout(dropout)

    def build(self, input_shape):
        super(SublayerConnection, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: List of two input tensors, x and a residual.
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        x, residual = x

        if training:
            pred = tf.random_uniform([]) < self.Pl
            return tf.cond(pred, lambda: residual, lambda: self.dropout(x) + residual)
        else:
            return x + residual

    def compute_output_shape(self, input_shape):
        return input_shape
