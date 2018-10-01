import tensorflow as tf
from src.models.utils import mask_logits


class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size=1, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(1,
                                           strides=1,
                                           use_bias=False,
                                           kernel_size=kernel_size)

    def call(self, block_1, block_2, mask=None):
        x = tf.concat([block_1, block_2], axis=-1)
        x = self.conv(x)
        x = tf.squeeze(x, axis=-1)
        x = mask_logits(x, mask)

        return x
