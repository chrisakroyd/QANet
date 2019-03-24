import tensorflow as tf
from tensorflow.keras.layers import Layer, Softmax


class PredictionHead(Layer):
    def __init__(self, answer_limit, **kwargs):
        """ Prediction Head

            The prediction head simply computes the 'best' valid start and end pointer using an upper triangular
            matrix.

            Args:
                answer_limit: Maximum answer length.
        """
        super(PredictionHead, self).__init__(**kwargs)
        self.answer_limit = answer_limit
        self.start_softmax = Softmax()
        self.end_softmax = Softmax()

    def call(self, x, training=None, mask=None):
        start_logits, end_logits = x
        lower = tf.cast(0, dtype=tf.int32)
        # Allows us to compute start/end pointers when we have a context length < answer limit.
        upper = tf.math.minimum(self.answer_limit, tf.shape(start_logits)[1])

        start_prob = self.start_softmax(start_logits)
        end_prob = self.end_softmax(end_logits)

        outer = tf.matmul(tf.expand_dims(start_prob, axis=2),
                          tf.expand_dims(end_prob, axis=1))

        outer = tf.linalg.band_part(outer, lower, upper)
        start_pointer = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
        end_pointer = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
        return start_prob, end_prob, start_pointer, end_pointer
