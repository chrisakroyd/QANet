import tensorflow as tf
from tensorflow.keras.layers import Dropout, Layer
from src import layers


class SublayerConnection(Layer):
    def __init__(self, layer, dropout, sublayer, total_sublayers, dropout_every=1, use_layer_dropout=True, **kwargs):
        """ SublayerConnection

            Every sublayer in QANet follows the rough template input -> LayerNorm -> Layer -> Dropout -> residual,
            this layer implements this common behaviour in a reuseable way. We follow the paper "QANet"
            (https://arxiv.org/pdf/1804.09541.pdf) and implement the LayerDropout regularization method from
            the paper "Deep Networks with Stochastic Depth" (https://arxiv.org/abs/1603.09382) which drops out entire
            layers with probability P that increases for layers deeper within the network.

            Args:
                layer: A Keras Layer object that is callable and supports masking.
                dropout: P of dropping a layer.
                sublayer: Integer or float value representing this layers position.
                total_sublayers: The total number of layers.
        """
        super(SublayerConnection, self).__init__(**kwargs)
        self.Pl = dropout * float(sublayer) / float(total_sublayers)
        self.layer_norm = layers.LayerNorm()
        self.given_layer = layer
        self.use_dropout = sublayer % dropout_every == 0
        self.use_layer_dropout = use_layer_dropout
        if self.use_dropout:
            self.dropout = Dropout(dropout)

    def build(self, input_shape):
        super(SublayerConnection, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: A input tensor of shape [batch_size, seq_length, ?].
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        residual = x

        if training and self.use_layer_dropout:
            pred = tf.random_uniform([]) < self.Pl
            return tf.cond(pred, lambda: residual, lambda: self.wrap_layer(x, training, mask) + residual)
        else:
            return self.wrap_layer(x, training, mask) + residual

    def wrap_layer(self, x, training=None, mask=None):
        """ Wraps the given layer in LayerNorm and Dropout.
            Args:
                x: A input tensor of shape [batch_size, seq_length, ?].
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        x = self.layer_norm(x)

        if self.given_layer.supports_masking:
            x = self.given_layer(x, training, mask)
        else:
            x = self.given_layer(x)

        if self.use_dropout:
            x = self.dropout(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
