import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv1D, Dropout, Layer
from src import layers


class MultiHeadAttention(Layer):
    def __init__(self, filters=128, num_heads=8, dropout=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.filters = filters
        # Linear mappings https://arxiv.org/pdf/1706.03762.pdf (Attention is all you Need) section 3.2
        self.queries_layer = Conv1D(self.filters, kernel_size=1, strides=1, use_bias=False)
        self.keys_layer = Conv1D(self.filters, kernel_size=1, strides=1, use_bias=False)
        self.values_layer = Conv1D(self.filters, kernel_size=1, strides=1, use_bias=False)
        self.output_layer = Conv1D(self.filters, kernel_size=1, strides=1, use_bias=False)
        # square root of key depth Attention is all you Need, 3.2.1
        self.depth = (self.filters // self.num_heads)
        self.scaling_factor = self.depth ** -0.5

        self.softmax = Activation('softmax')

        self.dropout = Dropout(dropout)

    def compute_input_shape(self, input_shape):
        shape = tf.shape(input_shape)
        return shape[0], shape[1]

    def split_heads(self, x, batch_size, length):
        # Split the last dimension + transpose result
        x = tf.reshape(x, shape=(batch_size, length, self.num_heads, self.depth))
        return tf.transpose(x, perm=(0, 2, 1, 3))

    def combine_heads(self, x, batch_size, length):
        x = tf.transpose(x, perm=(0, 2, 1, 3))  # --> [batch, length, num_heads, depth]
        return tf.reshape(x, shape=(batch_size, length, self.filters))

    def call(self, x, training=None, mask=None):
        x, y = x
        batch_size, length = self.compute_input_shape(x)
        query, key, values = (self.queries_layer(x), self.keys_layer(y), self.values_layer(y))
        # Split into n heads, allows model to jointly attend to different positions.
        query = self.split_heads(query, batch_size, length)
        key = self.split_heads(key, batch_size, length)
        values = self.split_heads(values, batch_size, length)
        # Query is scaled to prevent large dot products.
        query *= self.scaling_factor
        # Calculate the dot product attention for each head
        logits = tf.matmul(query, key, transpose_b=True)
        # Optionally apply a mask.
        if mask is not None:
            mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)  # reshape mask to [bs, 1, 1, num_heads]
            logits = layers.apply_mask(logits, mask)
        # Calculate attention weights + apply dropout.
        weights = self.softmax(logits)
        weights = self.dropout(weights, training=training)
        attention = tf.matmul(weights, values)
        # Recombine the heads + run result through output layer.
        attention = self.combine_heads(attention, batch_size, length)
        attention = self.output_layer(attention)

        return attention

    def compute_output_shape(self, input_shape):
        return input_shape
