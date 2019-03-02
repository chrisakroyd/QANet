import tensorflow as tf
from tensorflow.keras.layers import Dropout, Layer
from src import layers


class SublayerWrapper(Layer):
    def __init__(self, layer, dropout, sublayer, total_sublayers, dropout_every=1, use_layer_dropout=True, **kwargs):
        """ SublayerWrapper

            Every sublayer in QANet follows the rough template input -> LayerNorm -> Layer -> Dropout -> residual,
            this layer implements this common behaviour in a reuseable wrapper. We follow the paper "QANet"
            (https://arxiv.org/pdf/1804.09541.pdf) and implement the LayerDropout regularization method from
            the paper "Deep Networks with Stochastic Depth" (https://arxiv.org/abs/1603.09382) which drops out entire
            layers with probability P that increases linearly for layers deeper within the network.

            Args:
                layer: A Keras Layer object that is callable and supports masking.
                dropout: P of dropping a layer.
                sublayer: Integer or float value representing this layers position.
                total_sublayers: The total number of layers.
        """
        super(SublayerWrapper, self).__init__(**kwargs)
        self.layer_norm = layers.LayerNorm()
        self.given_layer = layer
        self.use_dropout = sublayer % dropout_every == 0

        self.use_layer_dropout = use_layer_dropout
        self.layer_survival_prob = 1 - (float(sublayer) / float(total_sublayers) * dropout)
        self.bernoulli = tf.distributions.Bernoulli(probs=self.layer_survival_prob, dtype=tf.float32)

        if self.use_dropout:
            self.dropout = Dropout(dropout)

    def build(self, input_shape):
        super(SublayerWrapper, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: A input tensor of shape [batch_size, seq_length, ?].
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        residual = x
        output = self.wrap_layer(x, training, mask)

        if self.use_layer_dropout:
            output_multiplier = tf.cond(training, lambda: self.bernoulli.sample(), lambda: self.layer_survival_prob)
            output = output * output_multiplier

        return output + residual

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
