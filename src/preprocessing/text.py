import re
import string
import unicodedata
from ftfy import fix_text

# Regexes
# Treat \t, \n and \r as whitespace despite being control characters.
whitespace_chars = {' ', '\t', '\n', '\r'}
articles = re.compile(r'\b(a|an|the)\b')
apostrophe = re.compile(r"('')")
apostrophe_like = re.compile(r'(``)')


def clean(text):
    """ Cleans text, fixing unicode errors, normalising quotes and stripping stray whitespace.
        Args:
            text: String text to be cleaned.
        Returns:
            Cleaned string.
    """
    text = fix_text(text)
    text = apostrophe.sub('" ', text)
    text = apostrophe_like.sub('" ', text)
    text = text.strip()

    text = list(text)
    # Normalize spaces + remove invalid characters.
    out_text = []
    for char in text:
        if is_invalid(char):
            continue

        if is_whitespace(char):
            out_text.append(' ')
        else:
            out_text.append(char)
    return ''.join(out_text)


def is_whitespace(char):
    """ Checks if the unicode character is a form of whitespace """
    cat = unicodedata.category(char)
    if char in whitespace_chars or cat == 'Zs':
        return True
    return False


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
