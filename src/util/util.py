import json
import os
from types import SimpleNamespace


def save_json(path, data):
    """ Saves data as a UTF-8 encoded .json file.
        Args:
            path: String path to a .json file.
            data: A dict or iterable.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def load_json(path):
    """ Loads a UTF-8 encoded .json file.
        Args:
            path: String path to a .json file.
        Returns:
            Loaded json as original saved type e.g. dict for index, list for saved lists.
    """
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def namespace_json(path):
    """ Turns a dict into an object, allows lookup via dot notation.
        Args:
            path: String path to a .json file.
        Returns:
            A namespace object.
    """
    return SimpleNamespace(**load_json(path))


def index_from_list(words, skip_zero=True):
    """ Turns a list of strings into a word: index lookup table.
        Args:
            words: A list of strings.
            skip_zero: Whether 0 should be skipped.
        Returns:
            A dict mapping words to integers.
    """
    return {word: (i+1) if skip_zero else i for i, word in enumerate(words)}


def make_dirs(directories):
    """ Creates non-existent directories.
        Args:
            directories: A string directory path or a list of directory paths.
    """
    if isinstance(directories, str):
        directories = [directories]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def load_vocab_files(paths):
    """ Loads a .json index as a list of words where each words position is its index.
        Args:
            paths: Iterable of string paths or string path pointing to .json word index file.
        Returns:
            A list of strings.
    """
    if isinstance(paths, str):
        paths = [paths]

    vocabs = []
    for path in paths:
        index = load_json(path)
        vocabs.append(sorted(index, key=index.get))
    return vocabs


def remove_keys(data, keys=[]):
    """ Removes specified keys from a list of dicts.
        Args:
            data: Iterable of dicts.
            keys: List of string keys to remove.
        Returns:
            Input data with keys removed.
    """
    for _, value in data.items():
        for key in keys:
            value.pop(key, None)
    return data
