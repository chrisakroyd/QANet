import numpy as np


def create_vocab(embedding_index):
    return set([e for e, _ in embedding_index.items()])


def generate_matrix(index, embedding_dimensions=300, skip_zero=True):
    if skip_zero:
        rows = len(index) + 1
    else:
        rows = len(index)
    matrix = np.random.normal(scale=0.1, size=(rows, embedding_dimensions))

    if skip_zero:
        matrix[0] = np.zeros(embedding_dimensions)

    return matrix


def zero_out_trainables(embedding_index, word_index, embedding_dimensions, trainable_words):
    for word in trainable_words:
        assert word in word_index
        # Zero out the vector for this word in the pre-trained index.
        embedding_index[word] = np.zeros((embedding_dimensions, ), dtype=np.float32)
    return embedding_index


def read_embeddings_file(path):
    embedding_index = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            values = line.strip().split(' ')
            # First line is num words + vector size when using fast text, we skip this.
            if i == 0 and len(values) == 2:
                print('Detected FastText vector format.')
                continue
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embedding_index[word] = coefs

    return embedding_index


def create_embedding_matrix(embedding_index, word_index, embedding_dimensions):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimensions))

    for word, index in word_index.items():
        if index > len(embedding_matrix):
            raise ValueError('Index larger than embedding matrix for {}'.format(word))

        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[index] = embedding_vector
            assert len(embedding_vector) == embedding_dimensions

    return embedding_matrix


def load_embedding(path,
                   word_index,
                   embedding_dimensions=300,
                   trainable_embeddings=[],
                   embedding_index=None):
    # Read the given embeddings file if its not given.
    embedding_index = embedding_index if embedding_index is not None else read_embeddings_file(path)

    if len(trainable_embeddings) > 0:
        embedding_index = zero_out_trainables(embedding_index, word_index,
                                              embedding_dimensions, trainable_embeddings)

    embedding_matrix = create_embedding_matrix(embedding_index, word_index, embedding_dimensions)

    return embedding_matrix


def load_embeddings(embedding_paths):
    word_embedding_path, trainable_embedding_path, char_embedding_path = embedding_paths
    print('Loading Embeddings...')
    word_matrix = np.load(word_embedding_path)
    trainable_matrix = np.load(trainable_embedding_path)
    character_matrix = np.load(char_embedding_path)
    return word_matrix, trainable_matrix, character_matrix
