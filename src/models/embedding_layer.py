import tensorflow as tf
from src.models.layers import HighwayLayer
Embedding = tf.keras.layers.Embedding


class EmbeddingLayer(tf.keras.Model):
    def __init__(self, word_matrix, trainable_matrix, character_matrix, filters, char_limit,
                 kernel_size=5, word_dim=300, char_dim=200, word_dropout=0.1, char_dropout=0.05,
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

        self.char_conv = tf.keras.layers.Conv1D(char_dim,
                                                kernel_size=kernel_size,
                                                activation='relu',
                                                name='char_embed_conv')

        self.max_pool = tf.keras.layers.GlobalMaxPool1D()

        self.word_embedding_dropout = tf.keras.layers.Dropout(word_dropout)
        self.character_embedding_dropout = tf.keras.layers.Dropout(char_dropout)

        self.highway_1 = HighwayLayer(word_dropout, layer_id=1)
        self.highway_2 = HighwayLayer(word_dropout, layer_id=2)
        # Need to down project for the shared model encoders.
        self.projection = tf.keras.layers.Conv1D(filters, kernel_size=1, strides=1, use_bias=False)

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
        char_embedding = self.max_pool(char_embedding)
        char_embedding = tf.reshape(char_embedding, shape=(-1, input_length, char_embedding.shape[-1],))

        trainable_embedding = self.trainable_embedding(self.trainable_tokens_mask(words))
        trainable_embedding = self.relu(trainable_embedding)

        word_embedding = tf.add(word_embedding, trainable_embedding)
        embedding = tf.concat([word_embedding, char_embedding], axis=2)

        # Change made so this would work in eager mode.
        embedding = self.highway_1(embedding)
        embedding = self.highway_2(embedding)

        embedding = self.projection(embedding)

        return embedding
