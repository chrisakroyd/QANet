import tensorflow as tf
from tensorflow.keras.layers import Layer


class PositionEncoding(Layer):
    def __init__(self, min_timescale=1.0, max_timescale=1.0e4, **kwargs):
        """ Position Encoding

            Follows paper "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf, section 3.3) and encodes
            each position in a sequence with unique sin/cosine wave to make up for the lack of recurrence.

            Args:
                min_timescale: Min value
                max_timescale: Max value
        """
        super(PositionEncoding, self).__init__(**kwargs)
        self.min_timescale = float(min_timescale)
        self.max_timescale = float(max_timescale)

    def compute_input_shape(self, x):
        shape = tf.shape(x)
        return shape[1], shape[2]

    def call(self, x, training=None, mask=None):
        length, channels = self.compute_input_shape(x)
        position = tf.cast(tf.range(length), dtype=tf.float32)
        num_timescales = channels // 2
        # Generate the signal with cos + sin waves.
        log_timescale_increment = (tf.log(self.max_timescale / self.min_timescale) / (tf.cast(num_timescales, dtype=tf.float32) - 1))
        inv_timescales = self.min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), dtype=tf.float32) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])
        # Add input and signal
        return x + signal

    def compute_output_shape(self, input_shape):
        return input_shape
