import tensorflow as tf
# useful link on pipelines: https://cs230-stanford.github.io/tensorflow-input-data.html


def tf_record_pipeline(filenames, params):
    """ Creates a dataset from a TFRecord file.
        Args:
            filenames: A list of paths to .tfrecord files.
            params: A dictionary of parameters.
        Returns:
            A `tf.data.Dataset` object.
    """
    int_feature = tf.FixedLenFeature([], tf.int64)
    str_feature = tf.FixedLenSequenceFeature([], tf.string, allow_missing=True)

    features = {
        'context_tokens': str_feature,
        'context_length': int_feature,
        'query_tokens': str_feature,
        'query_length': int_feature,
        'answer_starts': int_feature,
        'answer_ends': int_feature,
        'answer_id': int_feature,
    }

    def parse(proto):
        return tf.parse_single_example(proto, features=features)

    dataset = tf.data.TFRecordDataset(filenames,
                                      buffer_size=params.tf_record_buffer_size,
                                      num_parallel_reads=params.parallel_calls)

    dataset = dataset.map(parse, num_parallel_calls=params.parallel_calls)
    return dataset


def index_lookup(dataset, word_table, char_table, char_limit=16, num_parallel_calls=4):
    """ Adds a map function to the dataset that maps strings to indices.

        To save memory + hard drive space we store contexts and queries as tokenised strings. Therefore we need
        to perform two tasks; Extract characters and map words + chars to an index for the embedding layer.

        Args:
            dataset: A `tf.data.Dataset` object.
            word_table: A lookup table of string words to indices.
            char_table: A lookup table of string characters to indices.
            char_limit: Max number of characters per word.
            num_parallel_calls: An int for how many parallel lookups we perform.
        Returns:
            A `tf.data.Dataset` object.
    """

    def _lookup(fields):
        # +1 allows us to use 0 as a padding character without explicitly mapping it.
        context_words = word_table.lookup(fields['context_tokens']) + 1
        query_words = word_table.lookup(fields['query_tokens']) + 1
        # @TODO Revist the +1's -1 situation.
        # Get chars + lookup in table, as table is 0 indexed, we have the default at -1 for the pad which becomes 0
        # with the addition of 1 to again treat padding as 0 without needing to define a padding character.
        context_chars = tf.string_split(fields['context_tokens'], delimiter='')
        query_chars = tf.string_split(fields['query_tokens'], delimiter='')
        context_chars = tf.sparse_tensor_to_dense(char_table.lookup(context_chars), default_value=-1) + 1
        query_chars = tf.sparse_tensor_to_dense(char_table.lookup(query_chars), default_value=-1) + 1
        context_chars = context_chars[:, :char_limit]
        query_chars = query_chars[:, :char_limit]

        return {
            'context_words': tf.cast(context_words, dtype=tf.int32),
            'context_chars': tf.cast(context_chars, dtype=tf.int32),
            'context_length': tf.cast(fields['context_length'], dtype=tf.int32),
            'query_words': tf.cast(query_words, dtype=tf.int32),
            'query_chars': tf.cast(query_chars, dtype=tf.int32),
            'query_length': tf.cast(fields['query_length'], dtype=tf.int32),
            'answer_starts': tf.cast(fields['answer_starts'], dtype=tf.int32),
            'answer_ends': tf.cast(fields['answer_ends'], dtype=tf.int32),
            'answer_id': tf.cast(fields['answer_id'], dtype=tf.int32),
        }

    dataset = dataset.map(_lookup, num_parallel_calls=num_parallel_calls)
    return dataset


def create_buckets(params):
    """ Optionally generates bucket ranges if they aren't specified in the hparams.
        Args:
            params: A dictionary of parameters.
        Returns:
            A list of integers for the start of buckets.
    """
    # If no bucket ranges are explicitly defined, create using the bucket_size parameter
    if len(params.bucket_ranges) == 0:
        # Plus 1 as the bucket excludes the high number.
        return [i for i in range(0, params.context_limit + 1, params.bucket_size)]
    return params.bucket_ranges


def get_padded_shapes(params):
    """ Creates a dict of key: shape mappings for padding batches.
        Args:
            params: A dictionary of parameters.
        Returns:
            A dict mapping of key: shape
    """
    return {'context_words': [-1],
            'context_chars': [-1, params.char_limit],
            'context_length': [],
            'query_words': [-1],
            'query_chars': [-1, params.char_limit],
            'query_length': [],
            'answer_starts': [],
            'answer_ends': [],
            'answer_id': []}


def create_lookup_tables(word_vocab, char_vocab):
    """ Function that creates an index table for a word and character vocab.
        Args:
            word_vocab: A list of string words.
            char_vocab: A list of string characters.
        Returns:
            A lookup table for both the words and characters.
    """
    # default value is highest in the vocab as this is the OOV embedding, we generate non-zero indexed therefore -1.
    word_table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(word_vocab, dtype=tf.string),
                                                           default_value=len(word_vocab) - 1)
    char_table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(char_vocab, dtype=tf.string),
                                                           default_value=len(char_vocab) - 1)
    return word_table, char_table


def create_pipeline(params, word_table, char_table, record_paths, training=True):
    """ Function that creates an input pipeline for train/eval.

        Optionally uses bucketing to generate batches of a similar length. Output tensors
        are padded to the max within the batch.

        Args:
            params: A dictionary of parameters.
            word_table: A lookup table of string words to indices.
            char_table: A lookup table of string characters to indices.
            record_paths: A list of string filepaths for .tfrecord files.
            training: Boolean value signifying whether we are in train mode.
        Returns:
            A `tf.data.Dataset` object and an initializable iterator.
    """
    dataset = tf_record_pipeline(record_paths, params)
    dataset = dataset.cache().repeat()
    if training:
        dataset = dataset.shuffle(buffer_size=params.shuffle_buffer_size)
    # Perform word -> index mapping.
    dataset = index_lookup(dataset, word_table, char_table, char_limit=params.char_limit,
                           num_parallel_calls=params.parallel_calls)
    # We either bucket (used in paper, faster train speed) or just form batches padded to max.
    # Note: py_func doesn't return output shapes therefore we zero pad to the limits on each batch and slice to
    # the batch max during training. @TODO revisit and see if this can be avoided.
    padded_shapes = get_padded_shapes(params)
    if params.bucket and training:
        buckets = create_buckets(params)

        def length_fn(fields):
            return tf.cast(fields['context_length'], dtype=tf.int32)

        dataset = dataset.apply(
            tf.contrib.data.bucket_by_sequence_length(element_length_func=length_fn,
                                                      bucket_batch_sizes=[params.batch_size] * (len(buckets) + 1),
                                                      bucket_boundaries=buckets))
    else:
        dataset = dataset.padded_batch(
            batch_size=params.batch_size,
            padded_shapes=padded_shapes,
            drop_remainder=training
        )

    dataset = dataset.prefetch(buffer_size=params.max_prefetch)
    iterator = dataset.make_initializable_iterator()
    return dataset, iterator
