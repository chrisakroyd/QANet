import tensorflow as tf
from src import constants, layers, train_utils, util


class QANetContextual(tf.keras.Model):
    def __init__(self, embedding_matrix, char_matrix, trainable_matrix, params):
        """
            Contextual embedding variant of the QANet model that incorporates ELMO/BERT in both pre-processed and
            fine-tuning mode.
        """
        super(QANetContextual, self).__init__()
        self.global_step = tf.train.get_or_create_global_step()
        self.dropout = tf.placeholder_with_default(params.dropout, (), name='dropout')
        self.attn_dropout = tf.placeholder_with_default(params.attn_dropout, (), name='attn_dropout')
        self.low_memory = params.low_memory
        self.fixed_contextual = params.fixed_contextual_embeddings

        if not self.fixed_contextual:
            util.model_support_check(params.contextual_model)
            self.contextual_model = util.get_hub_module(params.contextual_model, trainable=True)

        self.embedding = layers.EmbeddingLayer(embedding_matrix, trainable_matrix, char_matrix,
                                               use_trainable=params.use_trainable, word_dim=params.embed_dim,
                                               char_dim=params.char_dim, word_dropout=self.dropout,
                                               char_dropout=self.dropout / 2, kernel_size=params.char_kernel_size,
                                               use_contextual=params.use_contextual)

        self.embedding_encoder = layers.EncoderBlockStack(blocks=params.embed_encoder_blocks,
                                                          conv_layers=params.embed_encoder_convs,
                                                          kernel_size=params.embed_encoder_kernel_width,
                                                          hidden_size=params.hidden_size,
                                                          heads=params.heads,
                                                          dropout=self.dropout,
                                                          attn_dropout=self.attn_dropout,
                                                          ff_inner_size=params.ff_inner_size,
                                                          recompute_gradients=params.low_memory,
                                                          name='embedding_encoder')

        self.context_query = layers.ContextQueryAttention(name='context_query_attention')

        self.model_encoder = layers.EncoderBlockStack(blocks=params.model_encoder_blocks,
                                                      conv_layers=params.model_encoder_convs,
                                                      kernel_size=params.model_encoder_kernel_width,
                                                      hidden_size=params.hidden_size,
                                                      heads=params.heads,
                                                      dropout=self.dropout,
                                                      attn_dropout=self.attn_dropout,
                                                      ff_inner_size=params.ff_inner_size,
                                                      recompute_gradients=params.low_memory,
                                                      name='model_encoder')

        self.start_output = layers.OutputLayer(name='start_logits', use_bias=params.output_bias)
        self.end_output = layers.OutputLayer(name='end_logits', use_bias=params.output_bias)

        self.predict_pointers = layers.PredictionHead(params.answer_limit)

    def call(self, x, training=True, mask=None):
        training = tf.cast(training, dtype=tf.bool)

        if self.fixed_contextual:
            context_words, context_chars, context_embedded, context_lengths, query_words, query_chars, query_embedded, \
            query_lengths = util.unpack_dict(x, keys=constants.PlaceholderKeys.FIXED_CONTEXTUAL_INPUTS)
        else:
            context_tokens, context_words, context_chars, context_lengths, query_tokens, query_words, query_chars, \
            query_lengths = util.unpack_dict(x, keys=constants.PlaceholderKeys.FINETUNE_CONTEXTUAL_INPUTS)

            context_embedded = self.contextual_model(inputs={'tokens': context_tokens, 'sequence_len': context_lengths},
                                                     signature='tokens', as_dict=True)['elmo']
            query_embedded = self.contextual_model(inputs={'tokens': query_tokens, 'sequence_len': query_lengths},
                                                   signature='tokens', as_dict=True)['elmo']

        context_mask = layers.create_mask(context_lengths, maxlen=tf.reduce_max(context_lengths))
        query_mask = layers.create_mask(query_lengths, maxlen=tf.reduce_max(query_lengths))

        # We pre-compute the float mask tensors once as this saves both memory and compute, in low-memory mode
        # this causes issues with variable scopes therefore we just use mask tensors.
        if not self.low_memory:
            context_attn_bias = layers.create_attention_bias(context_mask)
            query_attn_bias = layers.create_attention_bias(query_mask)
        else:
            context_attn_bias = context_mask
            query_attn_bias = query_mask

        context_emb = self.embedding([context_words, context_chars, context_embedded], training=training)
        query_emb = self.embedding([query_words, query_chars, query_embedded], training=training)

        context_enc = self.embedding_encoder(context_emb, training=training, mask=context_attn_bias)
        query_enc = self.embedding_encoder(query_emb, training=training, mask=query_attn_bias)

        c2q, q2c = self.context_query([context_enc, query_enc], training=training, mask=[context_mask, query_mask])

        # Input for the first stage of the model encoder, refer to section 2.2. of QANet paper for more details
        inputs = tf.concat([context_enc, c2q, context_enc * c2q, context_enc * q2c], axis=-1)

        enc_1 = self.model_encoder(inputs, training=training, mask=context_attn_bias)
        enc_2 = self.model_encoder(enc_1, training=training, mask=context_attn_bias)
        enc_3 = self.model_encoder(enc_2, training=training, mask=context_attn_bias)

        start_logits = self.start_output([enc_1, enc_2], training=training, mask=context_mask)
        end_logits = self.end_output([enc_1, enc_3], training=training, mask=context_mask)

        # Prediction head - Returns an int pointer to the start and end position of the answer segment on the context.
        start_prob, end_prob, start_pred, end_pred = self.predict_pointers([start_logits, end_logits])
        return start_logits, end_logits, start_pred, end_pred, start_prob, end_prob

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels, l2=None):
        start_loss = tf.losses.sparse_softmax_cross_entropy(logits=start_logits, labels=start_labels)
        end_loss = tf.losses.sparse_softmax_cross_entropy(logits=end_logits, labels=end_labels)
        loss = tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)

        if l2 is not None and l2 > 0.0:
            variables = tf.trainable_variables()
            variables = [v for v in variables if 'bias' not in v.name and 'scale' not in v.name]
            l2_loss = train_utils.l2_ops(l2, variables=variables)
            loss = loss + l2_loss

        return loss
