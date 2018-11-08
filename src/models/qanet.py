import tensorflow as tf
from src import layers, models, train_utils


class QANet(tf.keras.Model):
    def __init__(self, embedding_matrix, char_matrix, trainable_matrix, params):
        super(QANet, self).__init__()
        self.global_step = tf.train.get_or_create_global_step()

        self.embedding_block = models.EmbeddingLayer(embedding_matrix, trainable_matrix, char_matrix,
                                                     word_dim=params.embed_dim, char_dim=params.char_dim)

        self.embedding_encoder = models.EncoderBlockStack(blocks=params.embed_encoder_blocks,
                                                          conv_layers=params.embed_encoder_convs,
                                                          kernel_size=params.embed_encoder_kernel_width,
                                                          filters=params.filters,
                                                          heads=params.heads,
                                                          dropout=params.dropout,
                                                          ff_mul=params.feed_forward_multiplier,
                                                          name='embedding_encoder')

        self.context_query = layers.ContextQueryAttention(name='context_query_attention')

        self.model_encoder = models.EncoderBlockStack(blocks=params.model_encoder_blocks,
                                                      conv_layers=params.model_encoder_convs,
                                                      kernel_size=params.model_encoder_kernel_width,
                                                      filters=params.filters,
                                                      heads=params.heads,
                                                      dropout=params.dropout,
                                                      ff_mul=params.feed_forward_multiplier,
                                                      name='model_encoder')

        self.start_output = layers.OutputLayer(name='start_logits')
        self.end_output = layers.OutputLayer(name='end_logits')

        self.predict_pointers = layers.PredictionHead(params.answer_limit)

    def call(self, x, training=True, mask=None):
        context_words, context_chars, context_lengths, query_words, query_chars, query_lengths = x
        # Get the sequence length for this batch.
        context_max = tf.reduce_max(context_lengths)
        query_max = tf.reduce_max(query_lengths)
        # Init mask tensors on the trimmed input.
        context_mask = layers.create_mask(context_lengths, context_max)
        query_mask = layers.create_mask(query_lengths, query_max)

        # Embed the query + context
        context_emb = self.embedding_block([context_words, context_chars])
        query_emb = self.embedding_block([query_words, query_chars])

        # Encode the query + context.
        context_enc = self.embedding_encoder(context_emb, training=training, mask=context_mask)
        query_enc = self.embedding_encoder(query_emb, training=training, mask=query_mask)

        # Calculate the Context -> Query (c2q) and Query -> Context Attention (q2c).
        c2q, q2c = self.context_query([context_enc, query_enc], training=training, mask=[context_mask, query_mask])
        # Input for the first stage of the model encoder, refer to section 2.2. of QANet paper for more details
        inputs = tf.concat([context_enc, c2q, context_enc * c2q, context_enc * q2c], axis=-1)

        # Run through our stacked model encoder blocks on this representation.
        enc_1 = self.model_encoder(inputs, training=training, mask=context_mask)
        enc_2 = self.model_encoder(enc_1, training=training, mask=context_mask)
        enc_3 = self.model_encoder(enc_2, training=training, mask=context_mask)

        # Get the start/end logits from the output layers
        start_logits = self.start_output([enc_1, enc_2], training=training, mask=context_mask)
        end_logits = self.end_output([enc_1, enc_3], training=training, mask=context_mask)

        # Prediction head - Returns an int pointer to the start and end position of the answer segment on the context.
        start_pred, end_pred = self.predict_pointers([start_logits, end_logits])
        return start_logits, end_logits, start_pred, end_pred, c2q, q2c

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels, l2=None):
        start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=start_logits, labels=start_labels)

        end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=end_logits, labels=end_labels)

        loss = tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)

        if l2 is not None and l2 > 0.0:
            l2_loss = train_utils.l2_ops(l2)
            loss = loss + l2_loss

        return loss
