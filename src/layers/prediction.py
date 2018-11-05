import tensorflow as tf
from tensorflow.keras.layers import Layer, Softmax


class PredictionHead(Layer):
    def __init__(self, answer_limit, **kwargs):
        """ Prediction Head

            The prediction head simply computes the 'best' valid start and end pointer using an upper trainglular
            matrix.

            Args:
                answer_limit: Maximum answer length.
        """
        super(PredictionHead, self).__init__(**kwargs)
        self.answer_limit = answer_limit
        self.start_softmax = Softmax()
        self.end_softmax = Softmax()

    def build(self, input_shape):
        super(PredictionHead, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        start_logits, end_logits = x

        p_start = self.start_softmax(start_logits)
        p_end = self.end_softmax(end_logits)

        outer = tf.matmul(tf.expand_dims(p_start, axis=2),
                          tf.expand_dims(p_end, axis=1))

        outer = tf.matrix_band_part(outer, 0, self.answer_limit)
        start_pointer = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
        end_pointer = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
        return start_pointer, end_pointer

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1), (input_shape[-1][0], 1)
