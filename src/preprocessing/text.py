import re
import string
import unicodedata

# TODO: Revisit this and clean it up a bit.

# Regexes
# Treat \t, \n and \r as whitespace despite being control characters as well as zero-width characters.
whitespace_chars = {' ', '\t', '\n', '\r', '\u200b', '\u200c', '\u200d', '\ufeff', '\u200e'}
dash_chars = {'-'}
articles = re.compile(r'\b(a|an|the)\b')
apostrophe = re.compile(r"('')")
# Should filter out wiki style references e.g. [3], [123], [citation needed]
double_apostrophe = re.compile(r"('')")
apostrophe_like = re.compile(r'(`)')
multi_spaces = re.compile(r'\s{2,}')
elipsiss = re.compile(r'([.]{2,})')

space_before = re.compile(r'([:$\\])')

# Should filter out wiki style references e.g. [3], [123], [citation needed]
wiki_noise = re.compile(
    r'\[(((not)|(original)|(when)|(dubious)|(better)|(additional))(.*?)|(([Ff]ull )?(citation|verification|year|clarification) needed)|(vague)|(update)|(contradictory)|(specify)|(page needed)|(((by )|(according to ))?(who(m)?)\?)|((([Nn][Bb])|([Nn](ote)?|[Ww]eb))?\s\d+)|([A-Za-z]))\]')

double_punct = re.compile(r'(\w+[.,\/#!$%~\'\"^&\*;:{}=\-_`~()\[\]])([.,\/#!$%~\'\"^&\*;:{}=\-_`~()\[\]]\w+)')

space_before_paren = re.compile(r'(\w+|[^\w\s])(\((\w+))')
space_after_paren = re.compile(r'(\))(\w+)')


def text_span(text, spans, start_pointer, end_pointer):
    """ Given spans, text + start/end word pointers, extracts an answer span from the text. """
    start_char = spans[start_pointer][0]
    end_char = spans[end_pointer][-1]
    return text[start_char: end_char]


def normalize(text):
    """
        Normalizes unicode whitespace, dashes and invalid characters. As this does not modify the length or position
        of words it is considered non-destructive.
        Args:
            text: String text to be cleaned.
        Returns:
            Cleaned string.
    """
    text = text.strip()

    text = list(text)
    # Normalize spaces + remove invalid characters.
    out_text = []

    for char in text:
        if is_invalid(char):
            continue

        if is_whitespace(char):
            out_text.append(' ')
        elif is_dash(char):
            out_text.append(' - ')
        elif is_math_symbol(char):
            out_text.append(' {} '.format(char))
        else:
            out_text.append(char)

    return ''.join(out_text)


def clean(text):
    """ Cleans the given text by removing wikipedia noise ([citation needed], [1], etc.) recurring punctuation and
        multiple spaces. As this may significantly modify the string, any answer pointers will need to be updated
        before used for training.

        Args:
            text: String text to be cleaned.
        Returns:
            Cleaned string.
    """
    text = space_before.sub(r" \1", text)
    text = elipsiss.sub(r' \1 ', text)
    text = apostrophe_like.sub(r" \1", text)

    text = double_punct.sub(r'\1 \2', text)

    text = wiki_noise.sub('', text)

    text = space_before_paren.sub(r"\1 \2", text)
    text = space_after_paren.sub(r"\1 \2", text)

    text = text.strip()
    text = normalize(text)
    text = multi_spaces.sub(' ', text)

    return text


def is_whitespace(char):
    """ Checks if the unicode character is a form of whitespace """
    cat = unicodedata.category(char)
    if char in whitespace_chars or cat == 'Zs':
        return True
    return False


def is_dash(char):
    """ Checks if the unicode character is a dash character """
    cat = unicodedata.category(char)
    return char in dash_chars or cat == 'Pd'


def is_math_symbol(char):
    """ Checks if the unicode character is a math symbol """
    cat = unicodedata.category(char)
    return cat == 'Sm'


def is_invalid(char):
    char_int = ord(char)
    return char_int == 0 or char_int == 0xfffd


def normalize_answer(s):
    """ Normalizes answers, borrowed from SQuAD eval script. """
    def remove_articles(text):
        return articles.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
