import tensorflow as tf
import math


class CoordinateEmbedding(tf.keras.layers.Layer):
    def __init__(self, min_timescale=1.0, max_timescale=1.0e4, **kwargs):
        """ Coordinate Embedding

            Follows paper "Universal Transformers" (https://arxiv.org/pdf/1807.03819.pdf, section 2.1) and encodes
            each a sequence with unique sin/cosine wave for both word position and timestep.

            Args:
                min_timescale: Min value
                max_timescale: Max value
        """
        super(CoordinateEmbedding, self).__init__(**kwargs)
        self.min_timescale = float(min_timescale)
        self.max_timescale = float(max_timescale)

    def build(self, input_shape):
        self.channels = input_shape[0][-1]
        self.channel_padding = self.channels % 2
        self.num_timescales = self.channels // 2
        # We perform this op in python land as otherwise the same ops are re-used across while loop contexts which is
        # not allowed by tf.
        self.log_timescale_increment = (math.log(self.max_timescale / self.min_timescale) /
                                        max(float(int(self.num_timescales)) - 1, 1))
        # Generate the signal with cos + sin waves -> Pre-computed and stored for efficiency.
        super(CoordinateEmbedding, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        x_layer, step = x

        length = tf.shape(x_layer)[1]
        position = tf.cast(tf.range(length), dtype=tf.float32)

        inv_timescales = self.min_timescale * tf.exp(
            tf.cast(tf.range(self.num_timescales), dtype=tf.float32) * -self.log_timescale_increment)

        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        scaled_step = step * inv_timescales

        step_signal = tf.concat([tf.sin(scaled_step), tf.cos(scaled_step)], axis=0)
        pos_signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)

        signal = pos_signal + step_signal

        signal = tf.pad(signal, [[0, 0], [0, self.channel_padding]])
        signal = tf.reshape(signal, [1, length, self.channels])
        # Add input and signal
        return x_layer + signal
