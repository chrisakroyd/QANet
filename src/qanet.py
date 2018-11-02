import tensorflow as tf
from src.models import EmbeddingLayer, StackedEncoderBlocks, ContextQueryAttention, OutputLayer, PredictionHead
from src.models.utils import create_mask
from src import train_utils


class QANet(tf.keras.Model):
    def __init__(self, embedding_matrix, char_matrix, trainable_matrix, hparams):
        super(QANet, self).__init__()
        self.hparams = hparams
        self.global_step = tf.train.get_or_create_global_step()
        # Embedding layer for word + char embedding, with support for trainable embeddings.
        self.embedding_block = EmbeddingLayer(embedding_matrix, trainable_matrix, char_matrix,
                                              word_dim=self.hparams.embed_dim, char_dim=self.hparams.char_dim)
        # Shared embedding encoder between query + context
        self.embedding_encoder_blocks = StackedEncoderBlocks(blocks=self.hparams.embed_encoder_blocks,
                                                             conv_layers=self.hparams.embed_encoder_convs,
                                                             kernel_size=self.hparams.embed_encoder_kernel_width,
                                                             filters=self.hparams.filters,
                                                             heads=self.hparams.heads,
                                                             dropout=self.hparams.dropout,
                                                             ff_mul=self.hparams.feed_forward_multiplier,
                                                             name='embedding_encoder')
        # Context Query attention layer calculates two similarity matrices, one between context and query, and another
        # between query and context.
        self.context_query = ContextQueryAttention(name='context_query_attention')
        # Shared model encoder.
        self.model_encoder_blocks = StackedEncoderBlocks(blocks=self.hparams.model_encoder_blocks,
                                                         conv_layers=self.hparams.model_encoder_convs,
                                                         kernel_size=self.hparams.model_encoder_kernel_width,
                                                         filters=self.hparams.filters,
                                                         heads=self.hparams.heads,
                                                         dropout=self.hparams.dropout,
                                                         ff_mul=self.hparams.feed_forward_multiplier,
                                                         name='model_encoder')

        self.start_output = OutputLayer(name='start_logits')
        self.end_output = OutputLayer(name='end_logits')

        self.predict_pointers = PredictionHead(self.hparams.answer_limit)

    def init(self, placeholders, train):
        _, _, context_lengths, _, _, _, y_start, y_end, self.answer_id = placeholders
        # Init network
        start_logits, end_logits, self.start_pointer, self.end_pointer, _, _ = self.call(placeholders)

        self.loss = self.compute_loss(start_logits, end_logits, y_start, y_end, self.hparams.l2)

        if train:
            self.train_op = self.add_train_ops(self.hparams.learn_rate, self.hparams.gradient_clip)

        if train and 0.0 < self.hparams.ema_decay < 1.0:
            self.train_op, self.ema = train_utils.ema_ops(self.train_op, self.hparams.ema_decay)

    def add_train_ops(self, learn_rate, warmup_steps=1000, gradient_clip=5.0):
        self.lr = train_utils.inverse_exponential_warmup(learn_rate, warmup_steps)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                                beta1=self.hparams.beta1,
                                                beta2=self.hparams.beta2,
                                                epsilon=self.hparams.epsilon)

        if self.hparams.gradient_clip > 0.0:
            grads = self.optimizer.compute_gradients(self.loss)
            grads_and_vars = train_utils.clip_by_global_norm(grads, gradient_clip)
            train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        else:
            train_op = self.optimizer.minimize(self.loss, self.global_step)
        return train_op

    def slice_ops(self, words, chars, max):
        words = tf.slice(words, begin=(0, 0), size=(-1, max))
        chars = tf.slice(chars, begin=(0, 0, 0), size=(-1, max, self.hparams.char_limit))
        return words, chars

    def call(self, x, training=True):
        context_words, context_chars, context_lengths, query_words, query_chars, query_lengths, \
        y_start, y_end, answer_id = x
        # Trim the input sequences to the max non-zero length in batch (speeds up training).
        if self.hparams.dynamic_slice:
            # Calc the max length for the batch.
            context_max = tf.reduce_max(context_lengths)
            query_max = tf.reduce_max(query_lengths)
            context_words, context_chars = self.slice_ops(context_words, context_chars, context_max)
            query_words, query_chars = self.slice_ops(query_words, query_chars, query_max)
        else:
            context_max = self.hparams.context_limit
            query_max = self.hparams.query_limit
        # Init mask tensors on the trimmed input.
        context_mask = create_mask(context_lengths, context_max)
        query_mask = create_mask(query_lengths, query_max)
        # Embed the query + context
        context_emb = self.embedding_block([context_words, context_chars])
        query_emb = self.embedding_block([query_words, query_chars])

        # Encode the query + context.
        context_enc = self.embedding_encoder_blocks(context_emb, training=training, mask=context_mask)
        query_enc = self.embedding_encoder_blocks(query_emb, training=training, mask=query_mask)

        # Calculate the Context -> Query (c2q) and Query -> Context Attention (q2c).
        c2q, q2c = self.context_query([context_enc, query_enc], training=training, mask=[context_mask, query_mask])
        # Input for the first stage of the model encoder, refer to section 2.2. of QANet paper for more details
        inputs = tf.concat([context_enc, c2q, context_enc * c2q, context_enc * q2c], axis=-1)
        # Run through our stacked model encoder blocks on this representation.
        enc_1 = self.model_encoder_blocks(inputs, training=training, mask=context_mask)
        enc_2 = self.model_encoder_blocks(enc_1, training=training, mask=context_mask)
        enc_3 = self.model_encoder_blocks(enc_2, training=training, mask=context_mask)

        # Get the start/end logits from the output layers
        start_logits = self.start_output([enc_1, enc_2], training=training, mask=context_mask)
        end_logits = self.end_output([enc_1, enc_3], training=training, mask=context_mask)
        # Prediction head - Returns a pointer to the start and end position of the answer segment on the context.
        start_pointer, end_pointer = self.predict_pointers([start_logits, end_logits])
        return start_logits, end_logits, start_pointer, end_pointer, c2q, q2c

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels, l2):
        start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=start_logits, labels=start_labels)

        end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=end_logits, labels=end_labels)

        loss = tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)
        if l2 > 0.0:
            l2_loss = train_utils.l2_ops(l2)
            loss = loss + l2_loss

        return loss
