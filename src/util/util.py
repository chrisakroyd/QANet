import pathlib
import json
import os


def get_save_path(model, directory='./model_checkpoints', fold=None):
    model_name = model.__class__.__name__
    path = directory + '/{}/{}'.format(model_name, model_name)
    # create dirs if they don't exist.
    pathlib.Path(directory + '/{}/'.format(model_name)).mkdir(parents=True, exist_ok=True)

    if fold is not None:
        path = path + '-fold-{}'.format(fold)

    path = path + '.hdf5'

    return path


def save_json(path, index, format_json=True):
    with open(path, 'w', encoding='utf8') as f:
        text = json.dumps(index, sort_keys=True, indent=4, ensure_ascii=False)\
            if format_json else json.dumps(index, ensure_ascii=False)
        f.write(text)


def load_json(path):
    with open(path, encoding='utf8') as f:
        index = json.load(f)
        return index


def save_embeddings(path, embedding_matrix, word_index):
    with open(path, 'w', encoding='utf8') as embeddings:
        for key, value in word_index.items():
            embed = embedding_matrix[value, :]
            embeddings.write('%s %s\n' % (key, ' '.join(map(str, embed))))


def merge_embeddings(embedding_matrix, trainable_embedding_matrix, word_index, trainable_words):
    vocab_size = len(word_index) + 1
    num_trainable = len(trainable_words) + 1
    valid_word_range = vocab_size - num_trainable

    for word in trainable_words:
        index = word_index[word]
        # Merge the trainable and embedding matrix
        embedding_matrix[index] = trainable_embedding_matrix[index - valid_word_range]

    return embedding_matrix


# Generates a dict that acts a word_index for the trainable_words.
def index_from_list(words, add_one=True):
    index = {}

    for i, word in enumerate(words):
        index[word] = (i + 1) if add_one else i  # As word-index is 1 indexed.

    return index


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
