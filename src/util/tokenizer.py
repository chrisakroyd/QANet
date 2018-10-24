import numpy as np
import spacy
from collections import Counter
from nltk import word_tokenize

default_punct = set(list(' !"#$%&()*+,-./:;=@[\]^_`{|}~?'))


class Tokenizer:
    def __init__(self, lower=False, filters=default_punct, max_words=25000, max_chars=2500, min_word_occurrence=-1,
                 min_char_occurrence=-1, char_limit=16, vocab=None, word_index=None, char_index=None, oov_token='<oov>',
                 trainable_words=None, tokenizer='spacy'):
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
            # self.nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner', 'parser'])
            self.nlp = spacy.blank("en")
        elif self.tokenizer == 'nltk':
            self.nlp = word_tokenize
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
        if self.lower:
            text = text.lower()
        tokens = []
        for token in self.nlp(text):
            text = token.text if self.tokenizer == 'spacy' else token
            if text not in self.filters and len(text) > 0:
                tokens.append(text)

        return tokens

    def fit_on_texts(self, texts):
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

    # Takes in trainable words and max_features, limits word index to top features and adds the trainable words as high
    # id values to permit an add operation and trainable embeddings for selected tokens.
    def update_indexes(self):
        print('Total Words: %d' % len(self.word_counter))
        # Create an ordered list of words + chars below the max for each.
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
        # If we aren't given a vocab on initialisation, we update the vocab whenever called.
        if not self.given_vocab:
            self.vocab = set([word for word, _ in self.word_counter.items()])

    def set_vocab(self, vocab):
        self.vocab = set(vocab)
        self.given_vocab = True
        self.just_fit = True

    def init(self):
        # Word indexes haven't been initialised or need updating.
        if (len(self.word_index) == 0 or len(self.char_index)) == 0 or self.just_fit:
            # Add vocab + create indexes.
            self.update_vocab()
            self.update_indexes()
            self.just_fit = False

    def get_index_word(self, word):
        # Find common occurrences of the word if its not in other formats.
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in self.word_index:
                return self.word_index[each]
        return self.word_index[self.oov_token]

    def get_index_char(self, char):
        if char in self.char_index:
            return self.char_index[char]
        return self.char_index[self.oov_token]

    def pad_sequence(self, words, characters, seq_length):
        if len(words) < seq_length:
            pad_num = seq_length - len(words)
            words += [0] * pad_num
            characters += [[0] * self.char_limit] * pad_num
        return words, characters

    def tokens_to_sequences(self, tokens, seq_length, pad=False, numpy=True):
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
        # Wrap in list if string to avoid having to handle separately
        if not isinstance(texts, list):
            texts = [texts]
        self.init()
        tokens = [self.tokenize(str(text)) for text in texts]
        seq_words, seq_chars = self.tokens_to_sequences(tokens, seq_length, pad, numpy)
        return seq_words, seq_chars
