import tensorflow as tf

Embedding = tf.keras.layers.Embedding


class HighwayLayer(tf.keras.Model):
    def __init__(self, size, dropout, layer_id, bias=True):
        super(HighwayLayer, self).__init__()
        self.gate = tf.keras.layers.Conv1D(size,
                                           kernel_size=1,
                                           strides=1,
                                           use_bias=bias,
                                           padding='same',
                                           activation='sigmoid',
                                           name='gate_%d' % layer_id)

        self.trans = tf.keras.layers.Conv1D(size,
                                            kernel_size=1,
                                            strides=1,
                                            use_bias=bias,
                                            # activation='relu',
                                            activation=None,
                                            padding='same',
                                            name='activation_%d' % layer_id)

        # self.gate_dropout = tf.keras.layers.Dropout(rate=dropout)
        self.trans_dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, x, training=None, mask=None):
        gate_out = self.gate(x)
        # gate_out = self.gate_dropout(gate_out)
        trans_out = self.trans(x)
        trans_out = self.trans_dropout(trans_out)
        out = gate_out * trans_out + (1 - gate_out) * x
        return out


class EmbeddingLayer(tf.keras.Model):
    def __init__(self, word_matrix, trainable_matrix, character_matrix, filters, char_limit,
                 kernel_size=5, word_dim=300, char_dim=200, highway_layers=2, word_dropout=0.1, char_dropout=0.05,
                 mask_zero=True):
        super(EmbeddingLayer, self).__init__()
        self.char_dim = char_dim
        self.char_limit = char_limit
        self.filters = filters
        self.vocab_size = len(word_matrix)
        self.char_vocab_size = len(character_matrix)
        self.num_trainable = len(trainable_matrix)
        self.valid_word_range = self.vocab_size - self.num_trainable

        self.word_embedding = Embedding(input_dim=self.vocab_size,
                                        output_dim=word_dim,
                                        mask_zero=mask_zero,
                                        trainable=False,
                                        embeddings_initializer=tf.constant_initializer(word_matrix,
                                                                                       verify_shape=True),
                                        name='word_embedding')

        self.trainable_embedding = Embedding(input_dim=self.num_trainable,
                                             output_dim=word_dim,
                                             mask_zero=mask_zero,
                                             trainable=True,
                                             embeddings_initializer=tf.constant_initializer(trainable_matrix,
                                                                                            verify_shape=True),
                                             name='trainable_word_embedding')

        self.char_embedding = Embedding(input_dim=self.char_vocab_size,
                                        output_dim=char_dim,
                                        mask_zero=mask_zero,
                                        trainable=True,
                                        embeddings_initializer=tf.constant_initializer(character_matrix,
                                                                                       verify_shape=True),
                                        name='char_embedding')

        self.char_conv = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size, activation='relu',
                                                name='char_embed_conv')

        self.word_embedding_dropout = tf.keras.layers.Dropout(word_dropout)
        self.character_embedding_dropout = tf.keras.layers.Dropout(char_dropout)

        self.projection = tf.keras.layers.Conv1D(filters, kernel_size=1, strides=1, use_bias=False)
        # Change made so this would work in eager mode
        # self.highway_layers = tf.keras.models.Sequential([
        #     HighwayLayer(filters, dropout, layer_id=i) for i in range(highway_layers)
        # ])
        self.highway_1 = HighwayLayer(filters, word_dropout, layer_id=1)
        self.highway_2 = HighwayLayer(filters, word_dropout, layer_id=2)

        # Initialize the layers which masks the words into the range 0 to len(trainable_tokens), where 0 = Glove word,
        # values > 0 = trainable id's. Uses Relu to put them in range 0 - len(trainable_tokens)
        self.trainable_tokens_mask = tf.keras.layers.Lambda(lambda x: x - self.valid_word_range)
        self.relu = tf.keras.layers.Activation('relu')

    def call(self, x, training=None, mask=None):
        words, chars, input_length = x
        word_embedding = self.word_embedding(words)
        char_embedding = self.char_embedding(chars)

        char_embedding = tf.reshape(char_embedding, shape=(-1, self.char_limit, self.char_dim, ))

        word_embedding = self.word_embedding_dropout(word_embedding)
        char_embedding = self.character_embedding_dropout(char_embedding)

        char_embedding = self.char_conv(char_embedding)
        char_embedding = tf.reduce_max(char_embedding, axis=1)
        char_embedding = tf.reshape(char_embedding, shape=(-1, input_length, char_embedding.shape[-1],))

        trainable_embedding = self.trainable_embedding(self.trainable_tokens_mask(words))
        trainable_embedding = self.relu(trainable_embedding)

        word_embedding = tf.add(word_embedding, trainable_embedding)
        embedding = tf.concat([word_embedding, char_embedding], axis=2)

        embedding = self.projection(embedding)

        # Change made so this would work in eager mode.
        embedding = self.highway_1(embedding)
        embedding = self.highway_2(embedding)
        # embedding = self.highway_layers(embedding)

        return embedding
