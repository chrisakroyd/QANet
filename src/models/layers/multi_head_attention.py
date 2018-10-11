import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv1D, Dropout
from src.models.utils import split_last_dimension, combine_last_two_dimensions, mask_logits


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, filters=128, num_heads=8, dropout=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.filters = filters

        self.memory_conv = Conv1D(2 * self.filters,
                                  kernel_size=1,
                                  strides=1,
                                  name='memory_projection',
                                  use_bias=False)

        self.query_conv = Conv1D(self.filters,
                                 kernel_size=1,
                                 strides=1,
                                 name='query_projection',
                                 use_bias=False)

        # square root of key depth https://arxiv.org/pdf/1706.03762.pdf (Attention is all you Need)
        self.scaling_factor = (self.filters // self.num_heads) ** -0.5

        self.softmax = Activation('softmax')

        self.dropout = Dropout(dropout)

    def call(self, queries, training=None, mask=None):
        memory = queries
        memory = self.memory_conv(memory)
        query = self.query_conv(queries)

        Q = split_last_dimension(query, self.num_heads)
        K, V = [split_last_dimension(tensor, self.num_heads) for tensor in tf.split(memory, 2, axis=2)]

        # @TODO From Attention is all you need, scaling factor should apply to the result of matmul op(logits)
        Q *= self.scaling_factor

        logits = tf.matmul(Q, K, transpose_b=True)

        if mask is not None:
            mask = tf.reshape(mask, shape=[tf.shape(logits)[0], 1, 1, -1])
            logits = mask_logits(logits, mask)

        weights = self.softmax(logits)
        weights = self.dropout(weights, training=training)
        x = tf.matmul(weights, V)

        return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))
