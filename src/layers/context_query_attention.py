import tensorflow as tf
from tensorflow.keras.layers import Layer, Softmax
from src import layers


class ContextQueryAttention(Layer):
    def __init__(self, use_bias=True, **kwargs):
        """ Context-Query Attention implementation, also referred to as Attention-Flow layer.

            The Attention flow layer was introduced within "Bi-Directional Attention Flow for
            Machine Comprehension" (https://arxiv.org/pdf/1611.01603.pdf section 2.4) and
            is responsible for linking and fusing information both the context and query.
            This implementation uses DCN attention (https://arxiv.org/pdf/1611.01604.pdf) for
            calculating the context to query and query to context attention.

            We use a more memory efficient method to calculate the similarity matrix S than
            in the original implementation where S = W0[q, c, q*c], instead we break this down
            into 3 ops: W0[q], W1[c], W2[q*c] and S = W0[q] + W1[c] + W2[q*c].

            Args:
                use_bias: Whether or not to add a bias vector to the result.
        """
        super(ContextQueryAttention, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.query_activation = Softmax(axis=-1)
        self.context_activation = Softmax(axis=1)

    def compute_input_shape(self, x):
        """ Gets the first three dimensions of the input (batch_size, sequence length, hidden_units) """
        shape = tf.shape(x)
        return shape[0], shape[1], shape[2]

    def build(self, input_shape):
        """ Adds the necessary weights and an optional bias variable """
        hidden_size = int(input_shape[0][-1])

        self.W0 = self.add_weight(name='W0',
                                  shape=(hidden_size, 1),
                                  initializer='glorot_uniform',
                                  trainable=True)

        self.W1 = self.add_weight(name='W1',
                                  shape=(hidden_size, 1),
                                  initializer='glorot_uniform',
                                  trainable=True)

        self.W2 = self.add_weight(name='W2',
                                  shape=(1, 1, hidden_size),
                                  initializer='glorot_uniform',
                                  trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=[1],
                                        initializer='zero',
                                        trainable=True)

        super(ContextQueryAttention, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: List of two input tensors for the encoded context + query.
                training: Boolean flag for training mode.
                mask: Two boolean mask tensors, first for the context, second for query.
        """
        x_context, x_query = x
        context_mask, query_mask = mask
        batch_size, context_length, hidden_size = self.compute_input_shape(x_context)
        _, query_length, _ = self.compute_input_shape(x_query)
        mask_context = tf.expand_dims(context_mask, axis=2)
        mask_query = tf.expand_dims(query_mask, axis=1)

        query_reshape = tf.reshape(x_query, shape=(-1, hidden_size))
        context_reshape = tf.reshape(x_context, shape=(-1, hidden_size))
        # First construct a similarity matrix following method described in DCN paper.
        sub_mat_0 = tf.reshape(tf.matmul(query_reshape, self.W0), shape=(batch_size, query_length))
        sub_mat_0 = tf.tile(tf.expand_dims(sub_mat_0, axis=2), multiples=(1, 1, context_length))
        sub_mat_1 = tf.reshape(tf.matmul(context_reshape, self.W1), shape=(batch_size, context_length))
        sub_mat_1 = tf.tile(tf.expand_dims(sub_mat_1, axis=1), multiples=(1, query_length, 1))
        sub_mat_2 = tf.matmul(x_query * self.W2, tf.transpose(x_context, perm=(0, 2, 1)))
        # Add the matrices together and transpose to form a matrix of shape [bs, context_length, query_length]
        S = sub_mat_0 + sub_mat_1 + sub_mat_2

        if self.use_bias:
            S += self.bias

        S = tf.transpose(S, perm=(0, 2, 1))
        # Standard context to query attention.
        S_ = self.query_activation(layers.apply_mask(S, mask=mask_query))
        c2q = tf.matmul(S_, x_query)
        # DCN style query to context attention.
        S_T = tf.transpose(self.context_activation(layers.apply_mask(S, mask=mask_context)), perm=(0, 2, 1))
        q2c = tf.matmul(tf.matmul(S_, S_T), x_context)

        return c2q, q2c

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], (input_shape[0][-1] * 4)
