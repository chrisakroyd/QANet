import tensorflow as tf
from src.models.utils import mask_logits


class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size=1, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(1,
                                           strides=1,
                                           kernel_size=kernel_size)

    def build(self, input_shape):
        super(OutputLayer, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        block_1, block_2 = x
        x = tf.concat([block_1, block_2], axis=-1)
        x = self.conv(x)
        x = tf.squeeze(x, axis=-1)
        x = mask_logits(x, mask)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1]
