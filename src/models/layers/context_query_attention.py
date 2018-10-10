import tensorflow as tf
from src.models.utils import dot, batch_dot, mask_logits


class ContextQueryAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ContextQueryAttention, self).__init__(**kwargs)
        self.query_activation = tf.keras.layers.Softmax(axis=-1)
        self.context_activation = tf.keras.layers.Softmax(axis=1)

    def compute_input_shape(self, x):
        shape = tf.shape(x)
        return shape[1], shape[2]

    def build(self, input_shape):
        self.W0 = self.add_weight(name='W0',
                                  shape=(int(input_shape[0][-1]), 1),
                                  initializer='glorot_uniform',
                                  trainable=True)

        self.W1 = self.add_weight(name='W1',
                                  shape=(int(input_shape[1][-1]), 1),
                                  initializer='glorot_uniform',
                                  trainable=True)

        self.W2 = self.add_weight(name='W2',
                                  shape=(1, 1, int(input_shape[0][-1])),
                                  initializer='glorot_uniform',
                                  trainable=True)

        self.bias = self.add_weight(name='linear_bias',
                                    shape=[1],
                                    initializer='zero',
                                    trainable=True)
        super(ContextQueryAttention, self).build(input_shape)

    def call(self, x, training=None, mask=None):
        x_context, x_question = x
        context_mask, query_mask = mask
        context_length, _ = self.compute_input_shape(x_context)
        question_length, _ = self.compute_input_shape(x_question)
        mask_q = tf.expand_dims(query_mask, axis=1)
        mask_c = tf.expand_dims(context_mask, axis=2)

        subres0 = tf.tile(dot(x_context, self.W0), multiples=[1, 1, question_length])
        subres1 = tf.tile(tf.transpose(dot(x_question, self.W1), perm=(0, 2, 1)), multiples=[1, context_length, 1])
        subres2 = batch_dot(x_context * self.W2, tf.transpose(x_question, perm=(0, 2, 1)))
        S = subres0 + subres1 + subres2
        S += self.bias

        S_ = self.query_activation(mask_logits(S, mask=mask_q))
        S_T = tf.transpose(self.context_activation(mask_logits(S, mask=mask_c)), perm=(0, 2, 1))
        c2q = tf.matmul(S_, x_question)
        q2c = tf.matmul(tf.matmul(S_, S_T), x_context)

        return tf.concat([x_context, c2q, x_context * c2q, x_context * q2c], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], (input_shape[0][-1] * 4)
