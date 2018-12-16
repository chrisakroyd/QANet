import json
import os
import urllib.request
from io import BytesIO
from zipfile import ZipFile
from types import SimpleNamespace


def save_json(path, data, indent=None):
    """ Saves data as a UTF-8 encoded .json file.
        Args:
            path: String path to a .json file.
            data: A dict or iterable.
            indent: Pretty print the json with this level of indentation.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


def load_json(path):
    """ Loads a UTF-8 encoded .json file.
        Args:
            path: String path to a .json file.
        Returns:
            Loaded json as original saved type e.g. dict for index, list for saved lists.
    """
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def download_json(url, path):
    """ Downloads and saves a UTF-8 encoded .json file.
        Args:
            url: String url which points to a .json file.
            path: String path to save a .json file.
        Returns:
            Loaded json as original saved type e.g. dict for index, list for saved lists.
    """
    if not url.endswith('.json'):
        raise ValueError('Expected URL to be a .json file instead received {}'.format(url))
    req = urllib.request.Request(url)
    r = urllib.request.urlopen(req).read()
    cont = json.loads(r.decode('utf-8'))
    save_json(path, cont)
    return cont


def download_unpack_zip(url, path):
    """ Downloads and unpacks a zip file containing a single embedding file.

        TODO: Add in progress bar to keep user updated on huge embedding files. @cakroyd.

        Args:
            url: String url which points to a .zip file.
            path: String directory path into which we extract the zip.
        Returns:
            Filename of the downloaded embeddings.
    """
    resp = urllib.request.urlopen(url)

    with ZipFile(BytesIO(resp.read())) as zip_file:
        embedding_name = zip_file.namelist()[-1]
        zip_file.extractall(path)
        return embedding_name


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
