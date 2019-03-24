import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv1D, Dropout, Layer
from src import layers


class MultiHeadAttention(Layer):
    def __init__(self, hidden_size=128, num_heads=8, dropout=0.1, **kwargs):
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
                num_heads: Number of attention heads to compute, when num_heads == 1, multi-head attention == attention
                dropout: Fraction of units to drop.
                self_attention: Boolean value for whether to use self-attention on the inputs.
        """
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.queries_layer = Conv1D(self.hidden_size, kernel_size=1, use_bias=False,
                                    kernel_initializer=layers.create_initializer())

        self.keys_layer = Conv1D(self.hidden_size, kernel_size=1, use_bias=False,
                                 kernel_initializer=layers.create_initializer())

        self.values_layer = Conv1D(self.hidden_size, kernel_size=1, use_bias=False,
                                   kernel_initializer=layers.create_initializer())

        self.output_layer = Conv1D(self.hidden_size, kernel_size=1, use_bias=False,
                                   kernel_initializer=layers.create_initializer())

        if not self.hidden_size % self.num_heads == 0:
            raise ValueError('Hidden Size {} must be divisible by the number of attention heads {} with no remainder.'
                             .format(self.hidden_size, self.num_heads))

        # Square root of key depth Attention is all you Need, Section 3.2.1
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
        return tf.transpose(x, perm=(0, 2, 1, 3))  # --> [batch_size, num_heads, length, depth]

    def combine_heads(self, x, batch_size, length):
        """ Given a tensor, combines the attention heads and returns a tensor
            of shape [batch_size, seq_length, depth]
        """
        x = tf.transpose(x, perm=(0, 2, 1, 3))  # --> [batch_size, length, num_heads, depth]
        return tf.reshape(x, shape=(batch_size, length, self.hidden_size))

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: List of two input tensors of shape [batch_size, seq_length, units], if they are the same,
                   this is self-attention.
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        if isinstance(x, (tf.Tensor, tf.SparseTensor, tf.Variable)):  # Self attention -> X, Y = X, X
            x, y = x, x
        elif len(x) == 2:  # Attention between two tensors -> X, Y = X, Y
            x, y = x
        else:  # Invalid, we can only take a tensor or list of two tensors for attention.
            raise ValueError('Expected a maximum of two tensors passed to multi-head attention, got: {}'.format(len(x)))

        batch_size, length_x = self.compute_input_shape(x)
        _, length_y = self.compute_input_shape(y)
        query, key, values = self.queries_layer(x), self.keys_layer(y), self.values_layer(y)

        query = self.split_heads(query, batch_size, length_x)
        key = self.split_heads(key, batch_size, length_y)
        values = self.split_heads(values, batch_size, length_y)

        query *= self.scaling_factor
        logits = tf.matmul(query, key, transpose_b=True)

        if mask is not None:
            if mask.dtype == tf.bool:  # Boolean mask, we convert to be an attention bias vector + add.
                bias = layers.create_attention_bias(mask)  # [batch_size, 1, 1, length]
                logits = logits + bias
            elif mask.dtype == tf.float32:  # Mask is already a bias vector
                logits = logits + mask
            else:
                raise ValueError('Expected mask dtype to be tf.float32 or tf.bool')

        weights = self.softmax(logits)
        weights = self.dropout(weights, training=training)
        attention = tf.matmul(weights, values)

        attention = self.combine_heads(attention, batch_size, length_x)  # [batch_size, length_x, hidden_size]
        attention = self.output_layer(attention)

        return attention
