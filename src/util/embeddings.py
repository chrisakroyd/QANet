import numpy as np


def generate_matrix(index, embedding_dimensions=300, skip_zero=True):
    """ Function that zeroes out trainable words in the embedding index.

        For trainable embeddings we only use the embedding from the trainable matrix, to ensure that
        there is no pre-trained influence on these words we zero out the embedding in the pre-trained embedding
        index.

        Args:
            index: A dict of word to index.
            embedding_dimensions: Dimension of the embeddings.
            skip_zero: Whether or not 0 represents a padding character.
        Returns:
            Embedding index with any trainable words set to all zero.
    """
    if skip_zero:
        rows = len(index) + 1
    else:
        rows = len(index)
    matrix = np.random.normal(scale=0.1, size=(rows, embedding_dimensions))

    if skip_zero:
        matrix[0] = np.zeros(embedding_dimensions)

    return matrix


def zero_out_trainables(embedding_index, word_index, embedding_dimensions, trainable_words):
    """ Function that zeroes out trainable words in the embedding index.

        For trainable embeddings we only use the embedding from the trainable matrix, to ensure that
        there is no pre-trained influence on these words we zero out the embedding in the pre-trained embedding
        index.

        Args:
            embedding_index: A dict of word to embedding mappings.
            word_index: A dict of word to index mappings.
            embedding_dimensions: Dimension of the embeddings.
            trainable_words: A list of string keys for trainable words.
        Returns:
            Embedding index with any trainable words set to all zero.
    """
    for word in trainable_words:
        assert word in word_index
        # Zero out the vector for this word in the pre-trained index.
        embedding_index[word] = np.zeros((embedding_dimensions, ), dtype=np.float32)
    return embedding_index


def read_embeddings_file(path):
    """ Function for reading GloVe/FastText embeddings.
        Args:
            path: Path to the embeddings file.
        Returns:
            Embedding index mapping words to an n dimensional vector.
    """
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
    """ Converts the embedding index into an embedding matrix.
        Args:
            embedding_index: Path to the embeddings file.
            word_index: A dict of word: index mappings
            embedding_dimensions: Dimension of the output embeddings.
        Returns:
            A numpy matrix of shape [num_words + 1, embedding_dimension].
    """
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


def load_embedding_file(path, word_index, embedding_dimensions=300, trainable_embeddings=[], embedding_index=None):
    """ Function for reading embedding matrices saved with numpy.
        Args:
            path: A string path to a GLoVe/FastText formatted embedding file.
            word_index: A dict mapping of word: index
            embedding_dimensions: The dimension of the embeddings.
            trainable_embeddings: A list of words which are trainable. (e.g. OOV)
            embedding_index: A pre-loaded dict of word: vector mapping.
        Returns:
            A list of embedding matrices (In order of paths)
    """
    # Read the given embeddings file if its not given.
    embedding_index = embedding_index if embedding_index is not None else read_embeddings_file(path)

    if len(trainable_embeddings) > 0:
        embedding_index = zero_out_trainables(embedding_index, word_index,
                                              embedding_dimensions, trainable_embeddings)

    embedding_matrix = create_embedding_matrix(embedding_index, word_index, embedding_dimensions)

    return embedding_matrix


def load_numpy_files(paths):
    """ Function for reading embedding matrices saved with numpy.
        Args:
            paths: An iterable of string paths.
        Returns:
            A list of numpy matrices (In order of paths)
    """
    print('Loading Embeddings...')
    return [np.load(path) for path in paths]
