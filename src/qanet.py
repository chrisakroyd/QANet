import tensorflow as tf
from tensorflow.keras.layers import Conv1D
from src.models import EmbeddingLayer, StackedEncoderBlocks, ContextQueryAttention, OutputLayer, PredictionHead
from src.models.utils import create_mask


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
        # Trim the input sequences to the max non-zero length in batch (speeds up training).
        if self.hparams.dynamic_slice:
            # Calc the max length for the batch.
            context_max = tf.reduce_max(self.context_lengths)
            query_max = tf.reduce_max(self.query_lengths)
            self.slice_ops(context_max, query_max)
        else:
            context_max = self.hparams.context_limit
            query_max = self.hparams.query_limit
        # Init mask tensors on the trimmed input.
        self.context_mask = create_mask(self.context_lengths, context_max)
        self.query_mask = create_mask(self.query_lengths, query_max)
        # Init network
        self.step()

        if self.train and self.hparams.l2 > 0.0:
            self.add_l2_loss(self.hparams.l2)

        if self.train:
            self.add_train_ops(self.hparams.learn_rate)

        if self.train and self.hparams.ema_decay > 0.0:
            self.add_ema_ops()

    def add_train_ops(self, learn_rate):
        self.lr = tf.minimum(learn_rate, 0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                                beta1=self.hparams.beta1,
                                                beta2=self.hparams.beta2,
                                                epsilon=self.hparams.epsilon)

        if self.hparams.gradient_clip > 0.0:
            grads = self.optimizer.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            clipped_grads, _ = tf.clip_by_global_norm(gradients, self.hparams.gradient_clip)
            self.train_op = self.optimizer.apply_gradients(zip(clipped_grads, variables), global_step=self.global_step)
        else:
            self.train_op = self.optimizer.minimize(self.loss, self.global_step)

    def add_ema_ops(self):
        with tf.name_scope('ema_ops'):
            self.ema = tf.train.ExponentialMovingAverage(self.hparams.ema_decay)
            with tf.control_dependencies([self.train_op]):
                self.train_op = self.ema.apply(tf.trainable_variables() + tf.moving_average_variables())

    def add_l2_loss(self, l2):
        with tf.name_scope('l2_ops'):
            self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * l2
            self.loss += self.l2_loss

    def slice_ops(self, context_max, query_max):
        self.context_words = tf.slice(self.context_words, begin=(0, 0), size=(-1, context_max), name='context_slice')
        self.query_words = tf.slice(self.query_words, begin=(0, 0), size=(-1, query_max), name='query_slice')
        self.context_chars = tf.slice(self.context_chars, begin=(0, 0, 0), size=(-1, context_max,
                                                                                 self.hparams.char_limit))
        self.query_chars = tf.slice(self.query_chars, begin=(0, 0, 0), size=(-1, query_max, self.hparams.char_limit))

    def step(self):
        # Embed the query + context
        context_emb = self.embedding_block([self.context_words, self.context_chars])
        query_emb = self.embedding_block([self.query_words, self.query_chars])

        # Encode the query + context.
        context_enc = self.embedding_encoder_blocks(context_emb, training=self.train, mask=self.context_mask)
        query_enc = self.embedding_encoder_blocks(query_emb, training=self.train, mask=self.query_mask)

        # Calculate the Context -> Query (c2q) and Query -> Context Attention (q2c).
        self.c2q, self.q2c = self.context_query([context_enc, query_enc], training=self.train,
                                                mask=[self.context_mask, self.query_mask])
        # Input for the first stage of the model encoder, refer to section 2.2. of QANet paper for more details
        inputs = tf.concat([context_enc, self.c2q, context_enc * self.c2q, context_enc * self.q2c], axis=-1)
        # Run through our stacked model encoder blocks on this representation.
        enc_1 = self.model_encoder_blocks(inputs, training=self.train, mask=self.context_mask)
        enc_2 = self.model_encoder_blocks(enc_1, training=self.train, mask=self.context_mask)
        enc_3 = self.model_encoder_blocks(enc_2, training=self.train, mask=self.context_mask)

        # Get the start/end logits from the output layers
        start_logits = self.start_output([enc_1, enc_2], mask=self.context_mask)
        end_logits = self.end_output([enc_1, enc_3], mask=self.context_mask)

        # Prediction head - Returns a pointer to the start and end position of the answer segment on the context.
        self.start_pointer, self.end_pointer = self.predict_pointers([start_logits, end_logits])

        # Calc and sum losses.
        loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=start_logits, labels=self.y_start)

        loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=end_logits, labels=self.y_end)

        self.loss = tf.reduce_mean(loss_start + loss_end)
