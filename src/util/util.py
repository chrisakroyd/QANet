import json
import os
import urllib.request
from io import BytesIO
from collections import ChainMap
from zipfile import ZipFile
from types import SimpleNamespace
from src import constants, util


def save_json(path, data, indent=None):
    """ Saves data as a UTF-8 encoded .json file.
        Args:
            path: String path to a .json file.
            data: A dict or iterable.
            indent: Pretty print the json with this level of indentation.
    """
    assert isinstance(path, str) and len(path) > 0
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


def load_json(path):
    """ Loads a UTF-8 encoded .json file.
        Args:
            path: String path to a .json file.
        Returns:
            Loaded json as original saved type e.g. dict for index, list for saved lists.
    """
    assert isinstance(path, str) and len(path) > 0
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_multiple_jsons(paths):
    """ Loads multiple UTF-8 encoded .json file.
        Args:
            paths: List of string paths to .json files.
        Returns:
            Loaded json as original saved type e.g. dict for index, list for saved lists for each path.
    """
    assert isinstance(paths, list) or isinstance(paths, tuple)
    return [load_json(path) for path in paths]


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


def params_as_dict(params):
    """
        Converts all flags + values generated with the abseil-py flag (tf.flags) object into a flat python dict.
        Args:
            params: An instance of flags.Flags
        Returns:
            A flat dict of all flag parameters.
    """
    # Flags are segregated by the file they were defined in, reduce this down to a flat list of all our flags.
    flag_modules = [{v.name: v.value for v in values} for _, values in params.flags_by_module_dict().items()]
    full_flags = dict(ChainMap(*flag_modules))  # ChainMap == dictionary update within loop
    return full_flags


def file_exists(path):
    """ Tests whether or not a full filepath exists. """
    return os.path.exists(path)


def directory_exists(path):
    """ Tests whether or not a directory exists. """
    return os.path.isdir(path)


def directory_is_empty(path):
    """ Tests whether or not a directory is empty. """
    return len(os.listdir(path)) == 0


def load_config(params, path):
    """ Loads a config if it exists and there's existing checkpoints, otherwise alerts the user and returns params."""
    if file_exists(path):
        if len(os.listdir(os.path.dirname(path))) == 1:  # Only have a model config, check if this is intentional.
            if not util.yes_no_prompt(constants.Prompts.FOUND_CONFIG_NO_CHECKPOINTS.format(path=path)):
                os.remove(path)  # Delete the config.
                return params
        print('Using config for {run_name} found at {path}...'.format(run_name=params.run_name, path=path))
        return namespace_json(path)
    else:
        print('No existing config for {run_name}...'.format(run_name=params.run_name))
        return params


def save_config(params, path, overwrite=False):
    """ Saves abseil-py flag (tf.flags) object as a .json formatted file. """
    if not file_exists(path) or overwrite:
        save_json(path, params_as_dict(params), indent=2)


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


def unpack_dict(placeholder_dict, keys=None):
    """ Unpacks a dictionary into a tuple with the values in the same order as the keys given by keys param.
        Args:
            placeholder_dict: A dict of input tensors.
            keys: List of string keys representing the order placeholders should be returned.
        Returns:
            A tuple of input tensors.
    """
    if keys is None:
        raise ValueError('No keys given to unpack_dict.')

    return tuple([placeholder_dict[key] for key in keys if key in placeholder_dict])


def remove_keys(data, keys=None):
    """ Removes specified keys from a list of dicts.
        Args:
            data: A dictionary.
            keys: List of string keys to remove.
        Returns:
            Input data with keys removed.
    """
    if keys is None:
        raise ValueError('No keys passed to remove_keys.')

    for _, value in data.items():
        for key in keys:
            value.pop(key, None)

    return data


def pad_to_max_length(tokens, lengths):
    """ Pads a batch of tokenized strings to the max length within the batch.
        Args:
            tokens: A list of shape [num_rows, ?]
            lengths: A list of shape [num_rows]
        Returns:
            tokens with all rows padded to the max length within the batch [num_rows, max_len]
    """
    # If all tokens are the same length we don't need to do anything.
    if len(set(lengths)) == 1:
        return tokens
    elif len(tokens) != len(lengths):
        raise ValueError('tokens and lengths parameter are different shapes, first dimension should be equal.')

    max_len = max(lengths)

    for i in range(len(lengths)):
        if lengths[i] < max_len:
            assert len(tokens[i]) == lengths[i]
            tokens[i] = tokens[i] + [''] * (max_len - lengths[i])
            assert len(tokens[i]) == max_len

    return tokens


def filename(path):
    """ Extracts the filename from a given path.
        Args:
            path: A string filepath.
        Returns:
            Filename of the given path without extension.
    """
    base = os.path.basename(path)
    name = os.path.splitext(base)
    return name[0]
