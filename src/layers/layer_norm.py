import tensorflow as tf
from tensorflow.keras.layers import Layer


class LayerNorm(Layer):
    def __init__(self, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.epsilon = 1e-6

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(input_shape[-1], ),
                                     initializer=tf.ones_initializer(),
                                     trainable=True,
                                     name='layer_norm_scale')

        self.bias = self.add_weight(shape=(input_shape[-1], ),
                                    initializer=tf.zeros_initializer(),
                                    trainable=True,
                                    name='layer_norm_bias')
        super(LayerNorm, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        norm = (x - mean) * tf.rsqrt(variance + self.epsilon)
        return norm * self.scale + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape
