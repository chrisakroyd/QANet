import numpy as np
from .util import load_json


def create_vocab(embedding_index):
    vocab = set([e for e, _ in embedding_index.items()])
    return vocab


def generate_matrix(index, embedding_dimensions=300, skip_zero=True):
    rows = len(index) + 2 if skip_zero else len(index) + 1
    matrix = np.random.random_sample((rows, embedding_dimensions)) - 0.5
    return matrix


def generate_trainable_matrix(embedding_index, word_index, embedding_dimensions, trainable_words):
    embedding_matrix = np.zeros((len(trainable_words) + 1, embedding_dimensions))
    valid_word_range = len(word_index) - len(trainable_words)

    for word in trainable_words:
        if word in word_index:
            # Zero out the current vector for this word.
            embedding_index[word] = np.zeros((embedding_dimensions, ))
            # Randomly initialise the training word with values between -0.5 and 0.5
            embedding_matrix[(word_index[word] - valid_word_range)] = np.random.random_sample((embedding_dimensions, )) - 0.5
        else:
            print('{} not in word index, skipping...')

    return embedding_matrix, embedding_index


def read_embeddings_file(path):
    embedding_index = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            values = line.strip().split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs

    return embedding_index


def load_embedding_matrix(embedding_index, word_index, embedding_dimensions):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimensions))

    oov_count = 0
    for word, i in word_index.items():
        if i > len(embedding_matrix):
            break
        
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            # Give OOV's their own embedding.
            embedding_matrix[i] = embedding_index.get('<oov>')

            oov_count += 1
    return embedding_matrix, oov_count


def load_embeddings(path,
                    word_index,
                    embedding_dimensions=300,
                    trainable_embeddings=[],
                    embedding_index=None):
    print('Loading embeddings...')
    # Read the given embeddings file if its not given.
    embedding_index = embedding_index if embedding_index is not None else read_embeddings_file(path)

    if len(trainable_embeddings) > 0:
        trainable_matrix, embedding_index = generate_trainable_matrix(embedding_index,
                                                                      word_index,
                                                                      embedding_dimensions,
                                                                      trainable_embeddings)
    else:
        trainable_matrix = np.zeros((1, embedding_dimensions))

    embedding_matrix, oov_count = load_embedding_matrix(embedding_index, word_index, embedding_dimensions)

    return embedding_matrix, trainable_matrix


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


def load_contextual_embeddings(index_paths, embedding_paths, embed_dim, char_dim):
    word_index_path, trainable_index_path, char_index_path = index_paths
    word_embedding_path, trainable_embedding_path, char_embedding_path = embedding_paths

    print('Loading Contextual Embeddings...')

    word_index = load_json(word_index_path)
    trainable_word_index = load_json(trainable_index_path)
    char_index = load_json(char_index_path)

    word_matrix, _ = load_embeddings(word_embedding_path, word_index, embed_dim)
    trainable_matrix, _ = load_embeddings(trainable_embedding_path, trainable_word_index, embed_dim)
    character_matrix, _ = load_embeddings(char_embedding_path, char_index, char_dim)

    return word_matrix, trainable_matrix, character_matrix
