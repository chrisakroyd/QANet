import string
from collections import Counter
import spacy

default_punct = set(string.punctuation)


class Tokenizer(object):
    def __init__(self, lower=False, filters=default_punct, max_words=25000, max_chars=2500, min_word_occurrence=-1,
                 min_char_occurrence=-1, vocab=None, word_index=None, char_index=None, oov_token='<oov>',
                 trainable_words=None):
        """
            Constructs a Tokenizer Object, this is a wrapper around Spacy which tracks word/character usage as well
            as handling trainable words, unknown word tokens and filters.

            Args:
                lower: Whether to lower case all strings.
                filters: Set, list or string of any punctuation we should ignore.
                max_words: Maximum number of words to include in the word_index.
                max_chars: Maximum number of chars to include in the char_index.
                min_word_occurrence: How many times a word must occur to be included in the word_index.
                min_char_occurrence: How many times a char must occur to be included in the char_index.
                vocab: A list of strings that are considered our vocab e.g. list of words with embeddings.
                word_index: A dict of word: index mappings.
                char_index: A dict of char: index mappings.
                oov_token: Token to replace out of vocabulary words/chars with.
                trainable_words: A list of words which we want to have trainable embeddings.
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
        # Flag for if we have just run the fit_text or if we have a set vocab.
        self.just_fit = False
        self.given_vocab = vocab is not None

        self.nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner', 'parser'])

        if not isinstance(filters, set) and filters is not None:
            self.filters = set(filters)
        else:
            self.filters = set()

        self.init()

    def tokenize(self, text, error_correct=True):
        """ Splits a text or list of text into its constituent words.
            Args:
                text: string or list of string of untokenized text.
                error_correct: If we have a vocab, attempts to correct minor errors against it. e.g. Token is steve
                               but vocab only has Steve -> We replace steve with Steve. If in the vocab we do nothing.
            returns:
                List of tokenized text.
        """
        if self.lower:
            text = text.lower()

        tokens = []
        for token in self.nlp(text):
            text = token.text

            if self.given_vocab and error_correct and text not in self.vocab:
                # We generate a short list of candidate words
                for word in (text.lower(), text.capitalize(), text.lower().capitalize(), text.upper(), token.lemma_):
                    if word in self.vocab:
                        text = word
                        break

            if text not in self.filters and len(text) > 0:
                tokens.append(text)

        return tokens

    def fit_on_texts(self, texts, error_correct=True):
        """ Counts word/character occurrence.
            Args:
                texts: string or list of string of untokenized text.
                error_correct: If we have a vocab, attempts to correct minor errors against it. e.g. Token is steve
                               but vocab only has Steve -> We replace steve with Steve. If in the vocab we do nothing.
            returns:
                List of tokenized text.
        """
        tokenized = []
        if not isinstance(texts, list):
            texts = [texts]

        for text in texts:
            tokens = self.tokenize(text, error_correct)

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
            self.update_vocab()
            self.update_indexes()
            self.just_fit = False
