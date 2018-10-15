import tensorflow as tf
from tensorflow.keras.layers import Dropout, Layer


class LayerDropout(Layer):
    def __init__(self, dropout, sublayer, total_sublayers, **kwargs):
        super(LayerDropout, self).__init__(**kwargs)
        self.Pl = dropout * float(sublayer) / float(total_sublayers)
        self.dropout = Dropout(dropout)

    def build(self, input_shape):
        super(LayerDropout, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        x, residual = x

        if training:
            pred = tf.random_uniform([]) < self.Pl
            return tf.cond(pred, lambda: residual, lambda: self.dropout(x) + residual)
        else:
            return x + residual

    def compute_output_shape(self, input_shape):
        return input_shape
