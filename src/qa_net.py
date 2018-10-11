import tensorflow as tf
from tensorflow.keras.layers import Conv1D
from src.models import EmbeddingLayer, StackedEncoderBlocks, ContextQueryAttention, OutputLayer, PredictionHead


class QANet:
    def __init__(self, embedding_matrix, char_matrix, trainable_matrix, hparams, train=True):
        self.train = train
        self.hparams = hparams
        self.global_step = tf.train.get_or_create_global_step()
        # These are the model layers, Primarily think of this as 1 embedding encoder shared between
        # context + query -> context_attention -> 1 * encoder layers run 3 times with the first 2 outputs
        # being used to calc the start prob + last two outputs used to calc the end prob.
        # Optionally use elmo (requires tokenized text rather than index input).
        self.embedding_block = EmbeddingLayer(embedding_matrix, trainable_matrix, char_matrix,
                                              word_dim=self.hparams.embed_dim, char_dim=self.hparams.char_dim)

        # We map the (embed_dim + char_dim) dim representations to the dimension of hparams.filters.
        self.embed_projection = Conv1D(self.hparams.filters,
                                       kernel_size=1,
                                       strides=1,
                                       use_bias=False)

        self.embedding_encoder_blocks = StackedEncoderBlocks(blocks=self.hparams.embed_encoder_blocks,
                                                             conv_layers=self.hparams.embed_encoder_convs,
                                                             kernel_size=self.hparams.embed_encoder_kernel_width,
                                                             filters=self.hparams.filters,
                                                             heads=self.hparams.heads,
                                                             dropout=self.hparams.dropout,
                                                             name='embedding_encoder')
        # Context Query attention layer calculates two similarity matrices, one between context and query, and another
        # between query and context.
        self.context_query = ContextQueryAttention(name='context_query_attention')

        # Context query results in a 4 * filter (512 with standard hparams) vector at each position, paper does specify
        # that weights are shared, therefore we need to project this representation to hparams.filters.
        self.model_projection = Conv1D(self.hparams.filters,
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

        self.predict_pointers = PredictionHead(self.hparams.answer_limit)

    def init(self, placeholders):
        self.context_words, self.context_chars, self.question_words, self.question_chars, \
        self.y_start, self.y_end, self.answer_id = placeholders

        self.masks()
        self.step()

        if self.train:
            self.add_train_ops(self.hparams.learn_rate)

        if self.train and self.hparams.ema_decay > 0.0:
            self.add_ema_ops()

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
            clipped_grads, _ = tf.clip_by_global_norm(gradients, self.hparams.gradient_clip)
            self.train_op = self.optimizer.apply_gradients(
                    zip(clipped_grads, variables), global_step=self.global_step)
        else:
            self.train_op = self.optimizer.minimize(self.loss, self.global_step)

    def add_ema_ops(self):
        with tf.name_scope('ema_ops'):
            self.ema = tf.train.ExponentialMovingAverage(self.hparams.ema_decay, num_updates=self.global_step)
            with tf.control_dependencies([self.train_op]):
                self.train_op = self.ema.apply(tf.trainable_variables() + tf.moving_average_variables())

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
        # Trim the input sequences to the max non-zero length in batch (speeds up training).
        self.slice_ops()
        # Embed the question + context
        context_emb = self.embedding_block([self.context_words, self.context_chars])
        quextion_emb = self.embedding_block([self.question_words, self.question_chars])
        # Project down to hparams.filters at each position for the stacked embedding encoder blocks.
        context_emb = self.embed_projection(context_emb)
        quextion_emb = self.embed_projection(quextion_emb)

        # Encode the question + context with the embedding encoder
        context_encoded = self.embedding_encoder_blocks(context_emb, training=self.train, mask=self.context_mask)
        question_encoded = self.embedding_encoder_blocks(quextion_emb, training=self.train, mask=self.question_mask)

        # Calculate the Context -> Query (c2q) and Query -> Context Attention (q2c).
        self.c2q, self.q2c = self.context_query([context_encoded, question_encoded], training=self.train,
                                                mask=[self.context_mask, self.question_mask])
        # Input for the model encoder, refer to section 2.2. of QANet paper for more details
        inputs = tf.concat([context_encoded, self.c2q, context_encoded * self.c2q, context_encoded * self.q2c], axis=-1)

        # Down-project for the next block.
        self.enc = self.model_projection(inputs)
        # Run through our stack of stacked model encoder blocks on this representation.
        self.enc_1 = self.model_encoder_blocks(self.enc, training=self.train, mask=self.context_mask)
        self.enc_2 = self.model_encoder_blocks(self.enc_1, training=self.train, mask=self.context_mask)
        self.enc_3 = self.model_encoder_blocks(self.enc_2, training=self.train, mask=self.context_mask)

        # Get the start/end logits from the output layers
        start_logits = self.start_output([self.enc_1, self.enc_2], mask=self.context_mask)
        end_logits = self.end_output([self.enc_1, self.enc_3], mask=self.context_mask)

        # Prediction head - Returns a pointer to the start and end position of the answer segment on the context.
        self.start_pointer, self.end_pointer = self.predict_pointers([start_logits, end_logits])

        # Calc and sum losses.
        loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=start_logits, labels=self.y_start)

        loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=end_logits, labels=self.y_end)

        self.loss = tf.reduce_mean(loss_start + loss_end)

        # Add regularization loss over all trainable weights.
        if self.hparams.l2 > 0.0:
            with tf.name_scope('l2_ops'):
                self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.hparams.l2
                self.loss += self.l2_loss
        else:
            self.l2_loss = 0.0

        # Ensures that the ops are added to the graph.
        self.loss = tf.identity(self.loss)
        self.l2_loss = tf.identity(self.l2_loss)
