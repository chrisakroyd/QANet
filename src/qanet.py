import tensorflow as tf
from src.models import EmbeddingLayer, StackedEncoderBlocks, ContextQueryAttention, OutputLayer, PredictionHead
from src.models.utils import create_mask
from src import train_utils


class QANet:
    def __init__(self, embedding_matrix, char_matrix, trainable_matrix, hparams, train=True):
        self.train = train
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

    def init(self, placeholders):
        self.context_words, self.context_chars, self.context_lengths, self.query_words, self.query_chars, \
        self.query_lengths, self.y_start, self.y_end, self.answer_id = placeholders

        # Init network
        self.start_logits, self.end_logits, self.start_pointer, self.end_pointer = self.call(placeholders, self.train)

        self.loss, self.l2_loss = self.compute_loss(self.start_logits, self.end_logits, self.y_start, self.y_end,
                                                    self.hparams.l2)

        if self.train:
            self.train_op = self.add_train_ops(self.hparams.learn_rate, self.hparams.gradient_clip)

        if self.train and 0.0 < self.hparams.ema_decay < 1.0:
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

    def slice_ops(self, context_max, query_max, char_max):
        self.context_words = tf.slice(self.context_words, begin=(0, 0), size=(-1, context_max))
        self.query_words = tf.slice(self.query_words, begin=(0, 0), size=(-1, query_max))
        self.context_chars = tf.slice(self.context_chars, begin=(0, 0, 0), size=(-1, context_max, char_max))
        self.query_chars = tf.slice(self.query_chars, begin=(0, 0, 0), size=(-1, query_max, char_max))

    def call(self, x, training=True):
        # Trim the input sequences to the max non-zero length in batch (speeds up training).
        if self.hparams.dynamic_slice:
            # Calc the max length for the batch.
            context_max = tf.reduce_max(self.context_lengths)
            query_max = tf.reduce_max(self.query_lengths)
            self.slice_ops(context_max, query_max, self.hparams.char_limit)
        else:
            context_max = self.hparams.context_limit
            query_max = self.hparams.query_limit
        # Init mask tensors on the trimmed input.
        context_mask = create_mask(self.context_lengths, context_max)
        query_mask = create_mask(self.query_lengths, query_max)
        # Embed the query + context
        context_emb = self.embedding_block([self.context_words, self.context_chars])
        query_emb = self.embedding_block([self.query_words, self.query_chars])

        # Encode the query + context.
        context_enc = self.embedding_encoder_blocks(context_emb, training=training, mask=context_mask)
        query_enc = self.embedding_encoder_blocks(query_emb, training=training, mask=query_mask)

        # Calculate the Context -> Query (c2q) and Query -> Context Attention (q2c).
        self.c2q, self.q2c = self.context_query([context_enc, query_enc], training=training,
                                                mask=[context_mask, query_mask])
        # Input for the first stage of the model encoder, refer to section 2.2. of QANet paper for more details
        inputs = tf.concat([context_enc, self.c2q, context_enc * self.c2q, context_enc * self.q2c], axis=-1)
        # Run through our stacked model encoder blocks on this representation.
        enc_1 = self.model_encoder_blocks(inputs, training=training, mask=context_mask)
        enc_2 = self.model_encoder_blocks(enc_1, training=training, mask=context_mask)
        enc_3 = self.model_encoder_blocks(enc_2, training=training, mask=context_mask)

        # Get the start/end logits from the output layers
        start_logits = self.start_output([enc_1, enc_2], training=training, mask=context_mask)
        end_logits = self.end_output([enc_1, enc_3], training=training, mask=context_mask)
        # Prediction head - Returns a pointer to the start and end position of the answer segment on the context.
        start_pointer, end_pointer = self.predict_pointers([start_logits, end_logits])
        return start_logits, end_logits, start_pointer, end_pointer

    def compute_loss(self, start_logits, end_logits, y_start, y_end, l2):
        # Calc and sum losses.
        loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=start_logits, labels=y_start)

        loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=end_logits, labels=y_end)

        loss = tf.reduce_mean(loss_start) + tf.reduce_mean(loss_end)
        l2_loss = train_utils.l2_ops(l2)
        loss = loss + l2_loss

        return loss, l2_loss
