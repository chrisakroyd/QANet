import nltk
import numpy as np
import re
import spacy
from collections import Counter
from operator import itemgetter


default_punct = set(list(' !"#$%&()*+,-./:;=@[\]^_`{|}~?'))
default_oov_token = '<oov>'

multi_spaces = re.compile(r'\s+')


class Tokenizer:
    def __init__(self, vocab=None, word_index={}, char_index={}, max_words=25000, max_chars=2500,
                 oov_token=default_oov_token, filters=default_punct, lower=False, min_word_occurrence=-1,
                 min_char_occurrence=-1, trainable_words=[default_oov_token], tokenizer='spacy', use_chars=True):
        self.word_counter = Counter()
        self.char_counter = Counter()
        self.vocab = set(vocab)
        self.given_vocab = vocab is not None

        self.word_index = word_index
        self.char_index = char_index
        self.filters = filters
        self.max_words = max_words
        self.max_chars = max_chars
        self.oov_token = oov_token
        self.lower = lower
        self.trainable_words = trainable_words
        self.min_word_occurrence = min_word_occurrence
        self.min_char_occurrence = min_char_occurrence
        self.use_chars = use_chars
        self.tokenizer = tokenizer

        if self.tokenizer == 'spacy':
            self.nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner', 'parser'])
        elif self.tokenizer == 'nltk':
            self.nlp = nltk.word_tokenize
        else:
            raise ValueError('Unknown tokenizer scheme.')

        if not isinstance(self.filters, set):
            self.filters = set(filters)

        self.update_vocab()

    def tokenize(self, text):
        if self.lower:
            text = str(text).lower()

        if self.tokenizer == 'spacy':
            tokens = [token.text for token in self.nlp(text)
                      if token.text not in self.filters and len(token.text) > 0]
        else:
            tokens = [token for token in self.nlp(text)
                      if token not in self.filters and len(token) > 0]

        return tokens

    def fit_on_texts(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        for text in texts:
            tokens = self.tokenize(text)
            characters = [list(token) for token in tokens]

            for token, char_tokens in zip(tokens, characters):
                self.word_counter[token] += 1

                for char in char_tokens:
                    if char not in self.filters:
                        self.char_counter[char] += 1

    def update_indexes(self):
        # Create ordered dict in terms of popularity for chars + words, skipping index 0.
        sorted_words = sorted(self.word_counter.items())
        sorted_chars = sorted(self.char_counter.items())

        # Only update index if we have something to update.
        if len(sorted_words) > 0 and len(sorted_chars) > 0:
            self.word_index = {
                word: i + 1 for i, (word, count) in enumerate(sorted_words)
                if count > self.min_word_occurrence and (self.given_vocab and word in self.vocab)
            }
            self.char_index = {char: i + 1 for i, (char, count) in enumerate(sorted_chars)
                               if count > self.min_char_occurrence and char not in self.filters}
            # Add OOV token.
            self.word_index[self.oov_token] = len(self.word_index) + 1
            self.char_index[self.oov_token] = len(self.char_index) + 1

    # Takes in trainable words and max_features, limits word index to top features and adds the trainable words as high
    # id values to permit an add operation and trainable embeddings for selected tokens.
    def adjust_word_index(self, trainable_words=['<oov>']):
        new_word_index = {e: i + 1 for i, (e, _) in enumerate(self.word_index.items()) if i <= self.max_words}
        new_char_index = {e: i + 1 for i, (e, _) in enumerate(self.char_index.items()) if i <= self.max_chars}

        if len(trainable_words) > 0:
            trainable_word_index = {}
            trainable_word_set = set(trainable_words)
            last_assigned = 1

            # Shift every word that isnt trainable into the 1 to vocab_size - len(trainable_words) range
            for i, (key, _) in enumerate(sorted(new_word_index.items(), key=itemgetter(1))):
                if key not in trainable_word_set:
                    trainable_word_index[key] = last_assigned
                    last_assigned += 1

            # Add the trainable words onto the end of the index.
            for word in trainable_word_set:
                trainable_word_index[word] = last_assigned
                last_assigned += 1

            new_word_index = trainable_word_index

        if self.oov_token not in new_word_index:
            new_word_index[self.oov_token] = len(new_word_index) + 1  # Add OOV as the last character

        if self.oov_token not in new_char_index:
            new_char_index[self.oov_token] = len(new_char_index) + 1  # Add OOV as the last character

        self.word_index = new_word_index
        self.char_index = new_char_index

    def update_vocab(self):
        # Set our vocab
        if not self.vocab and not self.given_vocab:
            self.vocab = set([word for _, (word, _) in enumerate(self.word_index.items())])

    def get_index_word(self, word):
        if word in self.vocab:
            # Find common occurrences of the word if its not in other formats.
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in self.word_index:
                    return self.word_index[each]

        return self.word_index[self.oov_token]

    def get_index_char(self, char):
        for each in (char.lower(), char.upper()):
            if each in self.char_index:
                return self.char_index[each]

        return self.char_index[self.oov_token]

    def texts_to_sequences(self, texts, max_words=15, max_chars=16, numpy=True, pad=True):
        processed_words = []
        processed_chars = []
        unpadded_lengths = []

        # Wrap in list if string to avoid having to handle separately
        if isinstance(texts, str):
            texts = [texts]

        # Word indexes haven't been initialised yet.
        if len(self.word_index) == 0 or len(self.char_index) == 0:
            self.update_indexes()
            self.adjust_word_index(self.trainable_words)
            # Add vocab
            self.update_vocab()

        for text in texts:
            text = str(text)
            words, characters = [], []

            tokens = self.tokenize(text)
            unpadded_lengths.append(len(tokens))

            for token in tokens[:max_words]:
                words.append(self.get_index_word(token))

                if self.use_chars:
                    # Get all characters
                    index_chars = [self.get_index_char(char) for char in list(token)[:max_chars]]
                    # Pad to max characters
                    if len(index_chars) < max_chars and pad:
                        index_chars += [0] * (max_chars - len(index_chars))
                    characters.append(index_chars)

            # Pad to max words with 0.
            if len(words) < max_words and pad:
                pad_num = max_words - len(words)
                words += [0] * pad_num
                characters += [[0] * max_chars] * pad_num

            processed_words.append(words)

            if self.use_chars:
                processed_chars.append(characters)

        if numpy:
            processed_words = np.array(processed_words, dtype=np.int32)
            processed_chars = np.array(processed_chars, dtype=np.int32)

        return processed_words, processed_chars, unpadded_lengths