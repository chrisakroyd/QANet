import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv1D, Dropout, Layer
from src import layers


class MultiHeadAttention(Layer):
    def __init__(self, hidden_size=128, num_heads=8, dropout=0.1, self_attention=True, **kwargs):
        """ Multi-Headed Attention implementation.

            This is an implementation of multi-head attention based on the paper "Attention
            is all you Need" (https://arxiv.org/pdf/1706.03762.pdf section 3.2).

            We first project the tensor x with a linear layer to form the "query" tensor and then
            project the second tensor y to form the "key" and "value" tensors. Each of these tensors
            is then split to form tensors of [batch_size, num_heads, seq_length, depth]. The dot
            product of key + query is calculated, scaled and then softmaxed to obtain the
            attention probability distribution. Finally, the dot-product of these probabilities and values
            are calculated before the heads are re-combined and run through a final, linear
            output layer.

            Args:
                hidden_size: The number of units in the final dimension of the input tensor.
                num_heads: Number of attention heads to compute.
                dropout: Fraction of units to drop.
                self_attention: Boolean value for whether to use self-attention on the inputs.
        """
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.self_attention = self_attention
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        # Linear mappings
        self.queries_layer = Conv1D(self.hidden_size, kernel_size=1, use_bias=False)
        self.keys_layer = Conv1D(self.hidden_size, kernel_size=1, use_bias=False)
        self.values_layer = Conv1D(self.hidden_size, kernel_size=1, use_bias=False)
        self.output_layer = Conv1D(self.hidden_size, kernel_size=1, use_bias=False)
        # square root of key depth Attention is all you Need, 3.2.1
        self.depth = (self.hidden_size // self.num_heads)
        self.scaling_factor = self.depth ** -0.5

        self.softmax = Activation('softmax')

        self.dropout = Dropout(dropout)

    def compute_input_shape(self, input_shape):
        """ Gets the first two dimensions of the input (batch_size and sequence length) """
        shape = tf.shape(input_shape)
        return shape[0], shape[1]

    def split_heads(self, x, batch_size, length):
        """ Splits the given tensor and transposes the result, forming
            tensor of shape [batch_size, num_heads, seq_length, depth]
        """
        # Split the last dimension + transpose result
        x = tf.reshape(x, shape=(batch_size, length, self.num_heads, self.depth))
        return tf.transpose(x, perm=(0, 2, 1, 3))

    def combine_heads(self, x, batch_size, length):
        """ Given a tensor, combines the attention heads and returns a tensor
            of shape [batch_size, seq_length, depth]
        """
        x = tf.transpose(x, perm=(0, 2, 1, 3))  # --> [batch, length, num_heads, depth]
        return tf.reshape(x, shape=(batch_size, length, self.hidden_size))

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: List of two input tensors of shape [batch_size, seq_length, units], if they are the same,
                   this is self-attention.
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        if self.self_attention:
            x, y = x, x
        else:
            x, y = x
        # x, y = x
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