import json
import os
from types import SimpleNamespace


def save_json(path, index):
    with open(path, 'w') as f:
        json.dump(index, f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def namespace_json(path):
    return SimpleNamespace(**load_json(path))


# Generates a dict that acts a word_index for the trainable_words.
def index_from_list(words, add_one=True):
    return {word: (i+1) if add_one else i for i, word in enumerate(words)}


def concat_arrays(arrays):
    concatenated = []

    for arr in arrays:
        concatenated.extend(arr)
    return concatenated


def pad_array(array, limit):
    padded = array + ([0] * (limit - len(array)))
    return padded[:limit]


# Makes directories if they do not exist.
def make_dirs(directories):
    if isinstance(directories, str):
        directories = [directories]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def reverse_dict(dictionary):
    return {v: k for k, v in dictionary.items()}


def load_vocab(path):
    index = load_json(path)
    return sorted(index, key=index.get)
