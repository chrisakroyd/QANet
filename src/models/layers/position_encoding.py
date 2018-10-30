import tensorflow as tf
from tensorflow.keras.layers import Layer


class PositionEncoding(Layer):
    def __init__(self, min_timescale=1.0, max_timescale=1.0e4, **kwargs):
        super(PositionEncoding, self).__init__(**kwargs)
        self.min_timescale = float(min_timescale)
        self.max_timescale = float(max_timescale)

    def compute_input_shape(self, x):
        shape = tf.shape(x)
        return shape[1], shape[2]

    def call(self, x, training=None, mask=None):
        length, channels = self.compute_input_shape(x)
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2
        # Generate the signal with cos + sin waves.
        log_timescale_increment = (tf.log(self.max_timescale / self.min_timescale) / (tf.to_float(num_timescales) - 1))
        inv_timescales = self.min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])
        # Add input and signal
        return x + signal

    def compute_output_shape(self, input_shape):
        return input_shape