import tensorflow as tf
from src.models import EmbeddingLayer, StackedEncoderBlocks, ContextQueryAttention, OutputLayer


class QANet:
    def __init__(self, embedding_matrix, char_matrix, trainable_matrix, hparams, train=True):
        self.train = train
        self.hparams = hparams
        self.global_step = tf.train.get_or_create_global_step()
        # These are the model layers, Primarily think of this as 1 embedding encoder shared between
        # context + query -> context_attention -> 1 * encoder layers run 3 times with the first 2 outputs
        # being used to calc the start prob + last two outputs used to calc the end prob.
        # Optionally use elmo (requires tokenized text rather than index input.
        self.embedding_block = EmbeddingLayer(embedding_matrix, trainable_matrix, char_matrix,
                                              filters=self.hparams.filters, char_limit=self.hparams.char_limit,
                                              word_dim=self.hparams.embed_dim, char_dim=self.hparams.char_dim,
                                              mask_zero=False)

        self.embedding_encoder_blocks = StackedEncoderBlocks(blocks=self.hparams.embed_encoder_blocks,
                                                             conv_layers=self.hparams.embed_encoder_convs,
                                                             kernel_size=self.hparams.embed_encoder_kernel_width,
                                                             filters=self.hparams.filters,
                                                             heads=self.hparams.heads,
                                                             dropout=self.hparams.dropout,
                                                             name='embedding_encoder')

        self.context_query = ContextQueryAttention()

        self.input_projection = tf.keras.layers.Conv1D(self.hparams.filters,
                                                       strides=1,
                                                       use_bias=False,
                                                       kernel_size=1)

        self.model_encoder_blocks = StackedEncoderBlocks(blocks=self.hparams.model_encoder_blocks,
                                                         conv_layers=self.hparams.model_encoder_convs,
                                                         kernel_size=self.hparams.model_encoder_kernel_width,
                                                         filters=self.hparams.filters,
                                                         heads=self.hparams.heads,
                                                         dropout=self.hparams.dropout,
                                                         name='model_encoder')

        self.start_output = OutputLayer()
        self.end_output = OutputLayer()

        self.start_softmax = tf.keras.layers.Softmax()
        self.end_softmax = tf.keras.layers.Softmax()

    def init(self, placeholders):
        if self.hparams.use_elmo:
            self.context_words, self.question_words, self.y_start, self.y_end, self.answer_id = placeholders
        else:
            self.context_words, self.context_chars, self.question_words, self.question_chars, \
            self.y_start, self.y_end, self.answer_id = placeholders

        self.masks()
        self.step()

        if self.train:
            self.add_train_ops(self.hparams.learn_rate)

    def masks(self):
        # Initialize the masks
        self.context_mask = tf.cast(self.context_words, tf.bool)
        self.question_mask = tf.cast(self.question_words, tf.bool)
        self.context_length = tf.reduce_sum(tf.cast(self.context_mask, tf.int32), axis=1)
        self.question_length = tf.reduce_sum(tf.cast(self.question_mask, tf.int32), axis=1)

    def add_train_ops(self, learn_rate):
        self.lr = tf.minimum(learn_rate,  0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.8, epsilon=1e-7)

        if self.hparams.gradient_clip > 0.0:
            grads = self.optimizer.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(gradients, self.hparams.gradient_clip)
            self.train_op = self.optimizer.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)
        else:
            self.train_op = self.optimizer.minimize(self.loss, self.global_step)

    def slice_ops(self):
        self.context_max = tf.reduce_max(self.context_length)
        self.question_max = tf.reduce_max(self.question_length)
        self.context_words = tf.slice(self.context_words, [0, 0], [-1, self.context_max])
        self.question_words = tf.slice(self.question_words, [0, 0], [-1, self.question_max])
        self.context_mask = tf.slice(self.context_mask, [0, 0], [-1, self.context_max])
        self.question_mask = tf.slice(self.question_mask, [0, 0], [-1, self.question_max])
        self.context_chars = tf.slice(self.context_chars, [0, 0, 0], [-1, self.context_max, self.hparams.char_limit])
        self.question_chars = tf.slice(self.question_chars, [0, 0, 0], [-1, self.question_max, self.hparams.char_limit])

    def step(self):
        if self.hparams.use_elmo:
            # Run through the embedding block
            c_emb = self.embedding_block(self.context_words, self.context_length)
            q_emb = self.embedding_block(self.question_words, self.question_length)
        else:
            if self.hparams.limit_to_max:
                self.slice_ops()
            # Embed the question + context
            c_emb = self.embedding_block([self.context_words, self.context_chars, self.context_max])
            q_emb = self.embedding_block([self.question_words, self.question_chars, self.question_max])

        # Encode the question + context with the embedding encoder
        c = self.embedding_encoder_blocks(c_emb, training=self.train, mask=self.context_mask)
        q = self.embedding_encoder_blocks(q_emb, training=self.train, mask=self.question_mask)
        # Run context -> query attention over the context and the query
        inputs = self.context_query([c, q], self.context_max, self.question_max, self.context_mask, self.question_mask)
        # Down-project for the next block.
        self.enc = self.input_projection(inputs)

        self.enc_1 = self.model_encoder_blocks(self.enc, training=self.train, mask=self.context_mask)
        self.enc_2 = self.model_encoder_blocks(self.enc_1, training=self.train, mask=self.context_mask)
        self.enc_3 = self.model_encoder_blocks(self.enc_2, training=self.train, mask=self.context_mask)

        # Get the start/end logits from the output layers
        logits_start = self.start_output(self.enc_1, self.enc_2, mask=self.context_mask)
        logits_end = self.end_output(self.enc_1, self.enc_3, mask=self.context_mask)

        # Prediction head
        outer = tf.matmul(tf.expand_dims(self.start_softmax(logits_start), axis=2),
                          tf.expand_dims(self.end_softmax(logits_end), axis=1))

        outer = tf.matrix_band_part(outer, 0, self.hparams.answer_limit)
        self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
        self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

        # Loss calc
        loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits_start, labels=self.y_start)

        loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits_end, labels=self.y_end)

        self.loss = tf.reduce_mean(loss_start + loss_end)

        # Add regularization loss over all trainable weights.
        if self.hparams.l2_strength > 0.0:
            self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.hparams.l2_strength
            self.loss += self.l2_loss

        # EMA maintains a shadow copy of the trainable variables, increases memory usage as well as test performonce.
        if self.hparams.ema_decay > 0.0:
            self.var_ema = tf.train.ExponentialMovingAverage(self.hparams.ema_decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())

            with tf.control_dependencies([ema_op]):
                self.reg_loss = tf.identity(self.l2_loss)
                self.loss = tf.identity(self.loss)
                self.assign_vars = [tf.assign(var, self.var_ema.average(var)) for var in tf.trainable_variables()]
