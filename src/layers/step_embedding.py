import tensorflow as tf
from tensorflow.keras.layers import Layer


class StepEmbedding(Layer):
    def __init__(self, num_layers, **kwargs):
        """ Add Learned n-dimensional embedding as the step (vertical) timing signal.
            Adds embeddings to represent the position of the layer in the tower.

            Args:
                num_layers: Total number of layers.
        """
        super(StepEmbedding, self).__init__(**kwargs)
        self.num_layers = num_layers

    def build(self, input_shape):
        channels = int(input_shape[0][-1])
        self.step_embedding = self.add_weight(name='step_embedding',
                                              shape=(self.num_layers, 1, 1, channels),
                                              initializer=tf.random_normal_initializer(0, channels ** -0.5),
                                              trainable=True) * (channels ** 0.5)
        super(StepEmbedding, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        x_layer, step = x
        signal = self.step_embedding[step, :, :, :]
        return x_layer + signal
