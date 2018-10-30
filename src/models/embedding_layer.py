import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv1D, Dropout, Embedding
from src.models.layers import HighwayLayer


class EmbeddingLayer(tf.keras.Model):
    def __init__(self, word_matrix, trainable_matrix, character_matrix, kernel_size=5, word_dim=300,
                 char_dim=200, word_dropout=0.1, char_dropout=0.05, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.char_dim = char_dim
        self.vocab_size = len(word_matrix)
        self.char_vocab_size = len(character_matrix)
        self.num_trainable = len(trainable_matrix)
        self.valid_word_range = self.vocab_size - self.num_trainable

        self.word_embedding = Embedding(input_dim=self.vocab_size,
                                        output_dim=word_dim,
                                        mask_zero=True,
                                        trainable=False,
                                        embeddings_initializer=tf.constant_initializer(word_matrix,
                                                                                       verify_shape=True),
                                        name='word_embedding')

        self.trainable_embedding = Embedding(input_dim=self.num_trainable,
                                             output_dim=word_dim,
                                             mask_zero=True,
                                             trainable=True,
                                             embeddings_initializer=tf.constant_initializer(trainable_matrix,
                                                                                            verify_shape=True),
                                             name='trainable_word_embedding')

        self.char_embedding = Embedding(input_dim=self.char_vocab_size,
                                        output_dim=char_dim,
                                        mask_zero=True,
                                        trainable=True,
                                        embeddings_initializer=tf.constant_initializer(character_matrix,
                                                                                       verify_shape=True),
                                        name='char_embedding')

        self.char_conv = Conv1D(char_dim, kernel_size=kernel_size, activation='relu', padding='same', name='char_conv')

        self.word_dropout = Dropout(word_dropout)
        self.char_dropout = Dropout(char_dropout)

        self.highway_1 = HighwayLayer(word_dropout, name='highway_1')
        self.highway_2 = HighwayLayer(word_dropout, name='highway_2')
        # This relu is for facilitating the trainable embeddings.
        self.relu = Activation('relu')

    def call(self, x, training=None, mask=None):
        words, chars = x
        char_shape = tf.shape(chars)
        num_words, num_chars = char_shape[1], char_shape[2]
        word_embedding = self.word_embedding(words)  # [bs, len_words, embed_dim]
        char_embedding = self.char_embedding(chars)  # [bs, len_words, len_chars, char_dim]
        char_embedding = tf.reshape(char_embedding, shape=(-1, num_chars, self.char_dim,))
        word_embedding = self.word_dropout(word_embedding, training=training)
        char_embedding = self.char_dropout(char_embedding, training=training)
        # Treat each character as a channel + reduce to the max representation.
        char_embedding = self.char_conv(char_embedding)  # [bs, len_words, len_chars, char_dim]
        char_embedding = tf.reduce_max(char_embedding, axis=1)  # [bs, len_words, char_dim]
        char_embedding = tf.reshape(char_embedding, shape=(-1, num_words, self.char_dim,))
        # Create a tensor full of indexes between 0 and the total number of trainable words + embed.
        trainable_embedding = self.trainable_embedding(words - self.valid_word_range)
        trainable_embedding = self.relu(trainable_embedding)
        word_embedding = tf.add(word_embedding, trainable_embedding)
        # Concat the word + char embeddings to form a vector of embed_dim + char_dim at each position.
        embedding = tf.concat([word_embedding, char_embedding], axis=2)
        # Two highway layers then a projection to a lower dimensional input for the shared embedding encoder layers.
        embedding = self.highway_1(embedding, training=training)
        embedding = self.highway_2(embedding, training=training)

        return embedding
