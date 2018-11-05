import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Layer
from src import layers


class OutputLayer(Layer):
    def __init__(self, kernel_size=1, **kwargs):
        """ Output Layer

            Takes in two blocks and computes the output logits
            for either a start or end pointer via masked softmax.

            Args:
                kernel_size: Output units at each position.
        """
        super(OutputLayer, self).__init__(**kwargs)
        self.conv = Conv1D(1, strides=1, kernel_size=kernel_size, use_bias=False)

    def build(self, input_shape):
        super(OutputLayer, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: List of two same-shaped input tensors of shape [batch_size, seq_length, units]
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        block_1, block_2 = x
        x = tf.concat([block_1, block_2], axis=-1)
        x = self.conv(x)
        x = tf.squeeze(x, axis=-1)
        x = layers.apply_mask(x, mask)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1]
