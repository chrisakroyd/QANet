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

        # We use the same initialization as the self attention/feed forward, this init works better because of the
        # massive use of layer norm, if no layer norm its best to use a more standard weight init scheme.
        self.W0 = self.add_weight(name='W0',
                                  shape=(hidden_size, 1),
                                  initializer=layers.create_initializer(),
                                  trainable=True)

        self.W1 = self.add_weight(name='W1',
                                  shape=(hidden_size, 1),
                                  initializer=layers.create_initializer(),
                                  trainable=True)

        self.W2 = self.add_weight(name='W2',
                                  shape=(1, 1, hidden_size),
                                  initializer=layers.create_initializer(),
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

        query_reshape = tf.reshape(x_query, shape=(-1, hidden_size))
        context_reshape = tf.reshape(x_context, shape=(-1, hidden_size))
        # First construct a similarity matrix following method described in DCN paper.
        sub_mat_0 = tf.reshape(tf.matmul(query_reshape, self.W0), shape=(batch_size, query_length))
        sub_mat_0 = tf.tile(tf.expand_dims(sub_mat_0, axis=2), multiples=(1, 1, context_length))
        sub_mat_1 = tf.reshape(tf.matmul(context_reshape, self.W1), shape=(batch_size, context_length))
        sub_mat_1 = tf.tile(tf.expand_dims(sub_mat_1, axis=1), multiples=(1, query_length, 1))
        sub_mat_2 = tf.matmul(x_query * self.W2, x_context, transpose_b=True)

        # Add the matrices together and transpose to form a matrix of shape [bs, context_length, query_length]
        similarity_matrix = sub_mat_0 + sub_mat_1 + sub_mat_2

        if self.use_bias:
            similarity_matrix += self.bias

        similarity_matrix = tf.transpose(similarity_matrix, perm=(0, 2, 1))  # [batch_size, context_length, query_length]

        # We take our two mask tensors + form a mask of equal shape to the similarity matrix.
        mask_context = tf.expand_dims(context_mask, axis=2)  # [batch_size, context_length, 1]
        context_mask_tiled = tf.tile(mask_context, [1, 1, query_length])  # [batch_size, context_length, query_length]
        mask_query = tf.expand_dims(query_mask, axis=1)  # [batch_size, 1, query_length]
        query_mask_tiled = tf.tile(mask_query, [1, context_length, 1])  # [batch_size, context_length, query_length]
        # Combine the two boolean mask tensors into a singular mask of shape [batch_size, context_length, query_length]
        similarity_mask = tf.logical_and(query_mask_tiled, context_mask_tiled)
        similarity_matrix = layers.apply_mask(similarity_matrix, mask=similarity_mask)

        # Standard context to query attention.
        c2q_act = self.query_activation(similarity_matrix)
        c2q = tf.matmul(c2q_act, x_query)
        # DCN style query to context attention.
        q2c_act = self.context_activation(similarity_matrix)
        q2c = tf.matmul(tf.matmul(c2q_act, q2c_act, transpose_b=True), x_context)

        return c2q, q2c
