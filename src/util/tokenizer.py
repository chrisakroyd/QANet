import string
from collections import Counter
import numpy as np
import spacy

default_punct = set(string.punctuation)


class Tokenizer(object):
    def __init__(self, lower=False, filters=default_punct, max_words=25000, max_chars=2500, min_word_occurrence=-1,
                 min_char_occurrence=-1, char_limit=16, vocab=None, word_index=None, char_index=None, oov_token='<oov>',
                 trainable_words=None, tokenizer='spacy'):
        """Constructs a Tokenizer Object.
            Args:
                lower: Whether to lower case all strings.
                filters: Set, list or string of any punctuation we should ignore.
                max_words: Maximum number of words to include in the word_index.
                max_chars: Maximum number of chars to include in the char_index.
                min_word_occurrence: How many times a word must occur to be included in the word_index.
                min_char_occurrence: How many times a char must occur to be included in the char_index.
                char_limit: Max number of characters per word.
                vocab: A list of strings that are considered our vocab e.g. list of words with embeddings.
                word_index: A dict of word: index mappings.
                char_index: A dict of char: index mappings.
                oov_token: Token to replace out of vocabulary words/chars with.
                trainable_words: A list of words which we want to have trainable embeddings.
                tokenizer: The tokenizer we should use, supports spacy, nltk and whitespace split.
        """
        self.word_counter = Counter()
        self.char_counter = Counter()
        self.vocab = set(vocab if vocab else [])
        self.trainable_words = set(trainable_words if trainable_words else [])

        self.word_index = word_index if word_index else {}
        self.char_index = char_index if char_index else {}
        self.max_words = max_words
        self.max_chars = max_chars
        self.oov_token = oov_token
        self.lower = lower
        self.min_word_occurrence = min_word_occurrence
        self.min_char_occurrence = min_char_occurrence
        self.char_limit = char_limit
        self.tokenizer = tokenizer
        # Flag for if we have just run the fit_text or if we have a set vocab.
        self.just_fit = False
        self.given_vocab = vocab is not None

        if self.tokenizer == 'spacy':
            self.nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner', 'parser'])
        else:
            raise ValueError('Unknown tokenizer scheme.')

        if not isinstance(filters, set):
            if isinstance(filters, str):
                filters = list(filters)
            self.filters = set(filters if filters else [])
        else:
            self.filters = filters

        self.init()

    def tokenize(self, text):
        """ Splits a text or list of text into its constituent words.
            Args:
                text: string or list of string of untokenized text.
            returns:
                List of tokenized text.
        """
        if self.lower:
            text = text.lower()
        tokens = []
        for token in self.nlp(text):
            text = token.text if self.tokenizer == 'spacy' else token
            if text not in self.filters and len(text) > 0:
                tokens.append(text)

        return tokens

    def fit_on_texts(self, texts):
        """ Counts word/character occurrence.
            Args:
                texts: string or list of string of untokenized text.
            returns:
                List of tokenized text.
        """
        tokenized = []
        if not isinstance(texts, list):
            texts = [texts]

        for text in texts:
            tokens = self.tokenize(text)

            for token in tokens:
                self.word_counter[token] += 1
                for char in list(token):
                    self.char_counter[char] += 1

            tokenized.append(tokens)
        self.just_fit = True
        return tokenized

    def update_indexes(self):
        """ Creates word, character and handles trainable words.

            To facilitate trainable embeddings only for a subset of word and OOV tokens we need a special way of
            handling the word and char index creation. We create the word indexes as normal, but instead of assigning
            trainable word ids based on how often it occurs they are always assigned to the highest Id's. For details
            on why refer to docstrings in src/models/embedding_layer.
        """
        print('Total Words: %d' % len(self.word_counter))
        sorted_words = self.word_counter.most_common(self.max_words)
        sorted_chars = self.char_counter.most_common(self.max_chars)
        # Create list of words/chars that occur greater than min and are in the vocab or not filtered.
        word_index = [word for (word, count) in sorted_words if count > self.min_word_occurrence and
                      word in self.vocab and word not in self.filters and word not in self.trainable_words]

        print('Words in vocab: %d' % len(word_index))

        char_index = [char for (char, count) in sorted_chars if count > self.min_char_occurrence and
                      char not in self.filters]

        # Give each word id's in continuous range 1 to index length + convert to dict.
        word_index = {word: i for i, word in enumerate(word_index, start=1)}
        char_index = {char: i for i, char in enumerate(char_index, start=1)}
        # Add any trainable words to the end of the index (Therefore can exceed max_words)
        vocab_size = len(word_index)
        for i, word in enumerate(self.trainable_words, start=1):
            assert (vocab_size + i) not in word_index.values()
            word_index[word] = vocab_size + i
        # Add OOV token to the word + char index (Always have an OOV token).
        if self.oov_token not in word_index:
            assert len(word_index) + 1 not in word_index.values()
            word_index[self.oov_token] = len(word_index) + 1  # Add OOV as the last character

        if self.oov_token not in char_index:
            assert len(char_index) + 1 not in char_index.values()
            char_index[self.oov_token] = len(char_index) + 1  # Add OOV as the last character

        self.word_index = word_index
        self.char_index = char_index

    def update_vocab(self):
        """ If we don't have a vocab sets the vocab to all the words within the word index. """
        if not self.given_vocab:
            self.vocab = set([word for word, _ in self.word_counter.items()])

    def set_vocab(self, vocab):
        """ Sets the vocab and prevents it being automatically changed.
            Args:
                vocab: List of strings for words that are in the vocab (e.g. words with embeddings).
        """
        self.vocab = set(vocab)
        self.given_vocab = True
        self.just_fit = True

    def init(self):
        """ Initialises the vocab, word and char indices if they have not been set or need updating. """
        # Word indexes haven't been initialised or need updating.
        if (len(self.word_index) == 0 or len(self.char_index)) == 0 or self.just_fit:
            # Add vocab + create indexes.
            self.update_vocab()
            self.update_indexes()
            self.just_fit = False

    def get_index_word(self, word):
        """ Maps a word to an index if its in the index else the OOV token.
            Args:
                word: A word string.
            Returns:
                An int index or index of the OOV token.
        """
        # Find common occurrences of the word if its not in other formats.
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in self.word_index:
                return self.word_index[each]
        return self.word_index[self.oov_token]

    def get_index_char(self, char):
        """ Maps a character to an index if its in the index else the OOV token.
            Args:
                char: A character string.
            Returns:
                An int index or index of the OOV token.
        """
        if char in self.char_index:
            return self.char_index[char]
        return self.char_index[self.oov_token]

    def pad_sequence(self, words, chars, seq_length):
        """ Pads a word + character sequence to the given sequence length.
            Args:
                words: List of integers of shape [?].
                chars: List of integers of shape [?, ?].
                seq_length: Desired sequence length.
            Returns:
                List of word ints of shape [seq_length] and a list of char ints of shape [seq_length, char_limit]
        """
        if len(words) < seq_length:
            pad_num = seq_length - len(words)
            words += [0] * pad_num
            chars += [[0] * self.char_limit] * pad_num
        return words, chars

    def tokens_to_sequences(self, tokens, seq_length, pad=False, numpy=True):
        """ Converts lists of tokens into lists of integers.

            This function maps string tokens to their corresponding index id and optionally
            pads to the max length in the sequence. We always pad the character dimension to
            the limit specified during initialisation to make it much easier to handle during
            training/loading.

            Args:
                tokens: A list of tokens to convert to integers.
                seq_length: The max sequence length
                numpy: Whether or not to return a list or a numpy array.
                pad: Whether to pad each list of integers to the sequence length.
            Returns:
                List of word ints of shape [num_items, seq_length] and a list of
                char ints of shape [num_items, seq_length, char_limit]
        """
        seq_words, seq_chars = [], []
        self.init()
        if not isinstance(tokens[-1], list):
            tokens = [tokens]
        # Work through our pre-tokenized text.
        for text in tokens:
            words, characters = [], []
            for token in text:
                words.append(self.get_index_word(token))
                # Get all characters
                index_chars = [self.get_index_char(char) for char in list(token)]
                # Pad to max characters
                if len(index_chars) < self.char_limit and pad:
                    index_chars += [0] * (self.char_limit - len(index_chars))
                characters.append(index_chars)

            # Pad to max words with 0.
            if pad:
                words, characters = self.pad_sequence(words, characters, seq_length)
            # Add to the list and limit to the max.
            seq_words.append(words[:seq_length])
            seq_chars.append(characters[:seq_length])

        if numpy:
            seq_words = np.array(seq_words, dtype=np.int32)
            seq_chars = np.array(seq_chars, dtype=np.int32)

        return seq_words, seq_chars

    def texts_to_sequences(self, texts, seq_length, numpy=True, pad=True):
        """ Converts text/list of text into a list of integers.
            Args:
                texts: A string/list of strings to convert.
                seq_length: The max sequence length
                numpy: Whether or not to return a list or a numpy array.
                pad: Whether to pad each list of integers to the sequence length.
            Returns:
                List of word ints of shape [num_items, seq_length] and a list of
                char ints of shape [num_items, seq_length, char_limit]
        """
        # Wrap in list if string to avoid having to handle separately
        if not isinstance(texts, list):
            texts = [texts]
        self.init()
        tokens = [self.tokenize(str(text)) for text in texts]
        seq_words, seq_chars = self.tokens_to_sequences(tokens, seq_length, pad, numpy)
        return seq_words, seq_chars
