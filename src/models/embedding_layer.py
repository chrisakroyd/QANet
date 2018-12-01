import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv1D, Dropout, Embedding
from src import layers


class EmbeddingLayer(tf.keras.Model):
    def __init__(self, word_matrix, trainable_matrix, character_matrix, kernel_size=5, word_dim=300,
                 char_dim=200, word_dropout=0.1, char_dropout=0.05, **kwargs):
        """ Embedding layer that handles word, character and trainable word embeddings.

            The QANet paper (https://arxiv.org/pdf/1804.09541.pdf, section 2.2.1) refers to using
            word, character and trainable <UNK> tokens without giving details on how the trainable
            token is handled. Coupled with this, paper doesn't mention whether a regular or Separable
            convolution is used for the character embedding.

            The we treat word embeddings by simply embedding the token ids. We deviate from
            the probable implementation in QANet by reshaping the embedded characters to form
            a [batch_size, seq_length * char_limit] tensor before the char conv as this slightly improves
            train performance without impacting EM + F1 in a measurable way.

            Trainable embeddings are handled in two stages. During preprocessing we assign trainable words
            the highest word ID's, e.g. Given 40000 words and 2 trainable words, the two trainable words will
            have the two highest ids regardless of occurrence in the text. During embedding, we subtract the
            highest non-trainable word id (word_range) from the int32 tensor so that the only positive integers
            correspond to the trainable word ID's. We then use the ReLU function which thresholds at 0
            (negative numbers -> 0), leaving us with an int32 tensor of trainable word indexes. This tensor
            is then embedded and added to the output of the standard word_embedding.

            E.g. Given a vocab size of 400 words and two trainable words, the two trainable words have Id's
            of 400 and 401 (we skip zero) and a word_range of 399. For the sequence [42, 4, 400, 401, 89]
            we subtract 399 giving us [-357, -395, 1, 2, -310]. After ReLU this becomes [0, 0, 1, 2, 0]
            and can be embedded with the trainable matrix giving us both pre-trained and trainable embeddings.

            Args:
                word_matrix: A [vocab_size + 1, word_dim] matrix containing word embeddings.
                trainable_matrix: A [num_trainable + 1, word_dim] matrix containing trainable word embeddings.
                character_matrix: A [num_chars + 1, char_dim] matrix containing character embeddings.
                kernel_size: Width of the character convolution kernel.
                word_dim: An integer value for the dimension of word embeddings.
                char_dim: An integer value for the dimension of character embeddings.
                word_dropout: Fraction of units to drop from the word embedding.
                char_dropout: Fraction of units to drop from the char embedding.
        """
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.char_dim = char_dim
        self.vocab_size = len(word_matrix)
        self.char_vocab_size = len(character_matrix)
        self.num_trainable = len(trainable_matrix)
        self.word_range = self.vocab_size - self.num_trainable

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

        self.highway_1 = layers.HighwayLayer(word_dropout, name='highway_1')
        self.highway_2 = layers.HighwayLayer(word_dropout, name='highway_2')
        # This relu is for facilitating the trainable embeddings.
        self.relu = Activation('relu')

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: List of two int32 input tensors of shape [batch_size, seq_length]
                   and [batch_size, seq_length, num_chars].
                training: Boolean flag for training mode.
                mask: A boolean mask tensor.
        """
        words, chars = x
        char_shape = tf.shape(chars)
        num_words, num_chars = char_shape[1], char_shape[2]
        # @TODO Look into using a Conv2D or Separable Conv for character dropout.
        word_embedding = self.word_embedding(words)  # [bs, len_words, embed_dim]
        char_embedding = self.char_embedding(chars)  # [bs, len_words, len_chars, char_dim]
        char_embedding = tf.reshape(char_embedding, shape=(-1, num_chars, self.char_dim,))
        char_embedding = self.char_dropout(char_embedding, training=training)
        # Treat each character as a channel + reduce to the max representation.
        char_embedding = self.char_conv(char_embedding)  # [bs, len_words, len_chars, char_dim]
        char_embedding = tf.reduce_max(char_embedding, axis=1)  # [bs, len_words, char_dim]
        char_embedding = tf.reshape(char_embedding, shape=(-1, num_words, self.char_dim,))
        # Create a tensor full of indexes between 0 and the total number of trainable words + embed.
        trainable_embedding = self.trainable_embedding(words - self.word_range)
        trainable_embedding = self.relu(trainable_embedding)
        word_embedding = tf.add(word_embedding, trainable_embedding)
        word_embedding = self.word_dropout(word_embedding, training=training)
        # Concat the word + char embeddings to form a vector of embed_dim + char_dim at each position.
        embedding = tf.concat([word_embedding, char_embedding], axis=2)

        embedding = self.highway_1(embedding, training=training, mask=mask)
        embedding = self.highway_2(embedding, training=training, mask=mask)

        return embedding
