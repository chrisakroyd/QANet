import tensorflow as tf
from tensorflow.keras.layers import Layer


class LayerNorm(Layer):
    def __init__(self, axis=-1, epsilon=1e-6, **kwargs):
        """ Layer Normalization implementation
            Adds a Layer Normalization layer, based on the paper
            "Layer Normalization" (https://arxiv.org/abs/1607.06450).
        """
        super(LayerNorm, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        """ Adds the necessary weights. """
        self.scale = self.add_weight(shape=(input_shape[self.axis], ),
                                     initializer=tf.ones_initializer(),
                                     trainable=True,
                                     name='layer_norm_scale')

        self.bias = self.add_weight(shape=(input_shape[self.axis], ),
                                    initializer=tf.zeros_initializer(),
                                    trainable=True,
                                    name='layer_norm_bias')
        super(LayerNorm, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: An input tensor.
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        mean = tf.reduce_mean(x, axis=self.axis, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=self.axis, keepdims=True)
        norm = (x - mean) * tf.rsqrt(variance + self.epsilon)
        return norm * self.scale + self.bias
