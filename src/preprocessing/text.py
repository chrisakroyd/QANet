import re
import string
from ftfy import fix_text

# Regexes
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
    return text


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
