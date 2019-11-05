import tensorflow as tf
from tensorflow.keras.layers import Layer


class StepEncoding(Layer):
    def __init__(self, num_layers, num_channels, **kwargs):
        """ Add n-dimensional embedding as the step (vertical) timing signal.
            Adds embeddings to represent the position of the layer in the tower.

            Args:
                num_layers: Total number of layers.
        """
        super(StepEncoding, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.num_channels = num_channels

    def build(self, input_shape):
        self.step_embedding = self.add_weight(name='step_embedding',
                                              shape=(self.num_layers, 1, 1, self.num_channels),
                                              initializer=tf.random_normal_initializer(0, self.num_channels ** -0.5),
                                              trainable=True) * (self.num_channels ** 0.5)
        super(StepEncoding, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        x_layer, step = x
        signal = self.step_embedding[step, :, :, :]
        return x_layer + signal
