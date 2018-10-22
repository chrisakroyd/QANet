import re
import string
from ftfy import fix_text

# Regexes
articles = re.compile(r'\b(a|an|the)\b')
apostrophe = re.compile(r"('')")
apostrophe_like = re.compile(r'(``)')


def clean(text):
    text = fix_text(text)
    text = apostrophe.sub('" ', text)
    text = apostrophe_like.sub('" ', text)
    text = text.strip()
    return text


def normalize_answer(s):
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
