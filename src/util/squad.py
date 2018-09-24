import re
from tqdm import tqdm

# Regexes
apostrophe = re.compile(r"('')")
apostrophe_like = re.compile(r"(``)")


def extract_contexts(data):
    contexts = []

    for data in tqdm(data['data']):
        for question_answer in data['paragraphs']:
            context = question_answer['context']
            context = apostrophe.sub('" ', context)
            context = apostrophe_like.sub('" ', context)

            contexts.append(context)

    return contexts
