import os
import tensorflow as tf
# useful link on pipelines: https://cs230-stanford.github.io/tensorflow-input-data.html


def tf_record_pipeline(filenames, buffer_size=1024, num_parallel_calls=4):
    """ Creates a dataset from a TFRecord file.
        Args:
            filenames: A list of paths to .tfrecord files.
            buffer_size: Number of records to buffer.
            num_parallel_calls: How many functions we run in parallel.
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

    data = tf.data.TFRecordDataset(filenames,
                                   buffer_size=buffer_size,
                                   num_parallel_reads=num_parallel_calls)

    data = data.map(parse, num_parallel_calls=num_parallel_calls)
    return data


def index_lookup(data, tables, char_limit=16, num_parallel_calls=4, has_labels=True):
    """ Adds a map function to the dataset that maps strings to indices.

        To save memory + hard drive space we store contexts and queries as tokenised strings. Therefore we need
        to perform two tasks; Extract characters and map words + chars to an index for the embedding layer.

        @TODO This is a pretty ugly solution to support labelled/unlabelled modes, refactor target?

        Args:
            data: A `tf.data.Dataset` object.
            tables: A tuple of contrib.lookup tables mapping string words to indices and string characters to indices.
            char_limit: Max number of characters per word.
            num_parallel_calls: An int for how many parallel lookups we perform.
            has_labels: Include labels in the output dict.
        Returns:
            A `tf.data.Dataset` object.
    """
    word_table, char_table = tables

    def _lookup(fields):
        # +1 allows us to use 0 as a padding character without explicitly mapping it.
        context_words = word_table.lookup(fields['context_tokens']) + 1
        query_words = word_table.lookup(fields['query_tokens']) + 1
        # @TODO Revist the +1's -1 situation.
        # Get chars + lookup in table, as table is 0 indexed, we have the default at -1 for the pad which becomes 0
        # with the addition of 1 to again treat padding as 0 without needing to define a padding character.
        context_chars = tf.string_split(fields['context_tokens'], delimiter='')
        query_chars = tf.string_split(fields['query_tokens'], delimiter='')
        context_chars = tf.sparse.to_dense(char_table.lookup(context_chars), default_value=-1) + 1
        query_chars = tf.sparse.to_dense(char_table.lookup(query_chars), default_value=-1) + 1
        context_chars = context_chars[:, :char_limit]
        query_chars = query_chars[:, :char_limit]

        out_dict = {
            'context_words': tf.cast(context_words, dtype=tf.int32),
            'context_chars': tf.cast(context_chars, dtype=tf.int32),
            'context_length': tf.cast(fields['context_length'], dtype=tf.int32),
            'query_words': tf.cast(query_words, dtype=tf.int32),
            'query_chars': tf.cast(query_chars, dtype=tf.int32),
            'query_length': tf.cast(fields['query_length'], dtype=tf.int32),
        }

        if has_labels:
            out_dict.update({
                'answer_starts': tf.cast(fields['answer_starts'], dtype=tf.int32),
                'answer_ends': tf.cast(fields['answer_ends'], dtype=tf.int32),
                'answer_id': tf.cast(fields['answer_id'], dtype=tf.int32),
            })

        return out_dict

    data = data.map(_lookup, num_parallel_calls=num_parallel_calls)
    return data


def create_buckets(bucket_size, max_size, bucket_ranges=None):
    """ Optionally generates bucket ranges if they aren't specified in the hparams.
        Args:
            bucket_size: Size of the bucket.
            max_size: Maximum length of the thing we want to bucket.
            bucket_ranges: Pre-generated bucket ranges.
        Returns:
            A list of integers for the start of buckets.
    """
    if bucket_ranges is None or len(bucket_ranges) == 0:
        return [i for i in range(0, max_size + 1, bucket_size)]
    return bucket_ranges


def get_padded_shapes(max_context=-1, max_query=-1, max_characters=16, has_labels=True):
    """ Creates a dict of key: shape mappings for padding batches.

        Args:
            max_context: Max size of the context, -1 to pad to max within the batch.
            max_query: Max size of the query, -1 to pad to max within the batch.
            max_characters: Max number of characters, -1 to pad to max within the batch.
            has_labels: Include padded shape for answer_starts and answer_ends.
        Returns:
            A dict mapping of key: shape
    """
    shape_dict = {
        'context_words': [max_context],
        'context_chars': [max_context, max_characters],
        'context_length': [],
        'query_words': [max_query],
        'query_chars': [max_query, max_characters],
        'query_length': []}

    if has_labels:
        shape_dict.update({
            'answer_starts': [],
            'answer_ends': [],
            'answer_id': []
        })
    return shape_dict


def create_lookup_tables(vocabs):
    """ Function that creates an index table for a word and character vocab, currently only works
        for vocabs without an explicit <PAD> character.
        Args:
            vocabs: List of strings representing a vocab for a table, string order in list determines lookup index.
        Returns:
            A lookup table for each vocab.
    """
    tables = []
    for vocab in vocabs:
        table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(vocab, dtype=tf.string),
                                                          default_value=len(vocab) - 1)
        tables.append(table)
    return tables


def create_pipeline(params, tables, record_paths, training=True):
    """ Function that creates an input pipeline for train/eval.

        Optionally uses bucketing to generate batches of a similar length. Output tensors
        are padded to the max within the batch.

        Args:
            params: A dictionary of parameters.
            tables: A tuple of contrib.lookup tables mapping string words to indices and string characters to indices.
            record_paths: A list of string filepaths for .tfrecord files.
            training: Boolean value signifying whether we are in train mode.
        Returns:
            A `tf.data.Dataset` object and an initializable iterator.
    """
    parallel_calls = get_num_parallel_calls(params)

    data = tf_record_pipeline(record_paths, params.tf_record_buffer_size, parallel_calls)
    data = data.cache()
    if training:
        data = data.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=params.shuffle_buffer_size))
    else:
        data = data.repeat()
    # Perform word -> index mapping.
    data = index_lookup(data, tables, char_limit=params.char_limit,
                        num_parallel_calls=parallel_calls)

    if params.bucket and training:
        buckets = create_buckets(params.bucket_size, params.max_tokens, params.bucket_ranges)

        def length_fn(fields):
            return fields['context_length']

        data = data.apply(
            tf.data.experimental.bucket_by_sequence_length(element_length_func=length_fn,
                                                           bucket_batch_sizes=[params.batch_size] * (len(buckets) + 1),
                                                           bucket_boundaries=buckets))
    else:
        padded_shapes = get_padded_shapes(max_characters=params.char_limit)
        data = data.padded_batch(
            batch_size=params.batch_size,
            padded_shapes=padded_shapes,
            drop_remainder=training
        )

    data = data.prefetch(buffer_size=params.max_prefetch)
    iterator = data.make_initializable_iterator()
    return data, iterator


def create_demo_pipeline(params, tables, data):
    """ Function that creates an input pipeline for demo mode, .

        Output tensors are padded to the max within the batch.

        Args:
            params: A dictionary of parameters.
            tables: A tuple of contrib.lookup tables mapping string words to indices and string characters to indices.
            data: A dictionary containing keys for context_tokens, context_length, query_tokens, query_length and
                  answer_id.
        Returns:
            A `tf.data.Dataset` object and an initializable iterator.
    """
    parallel_calls = get_num_parallel_calls(params)

    data = tf.data.Dataset.from_tensor_slices(dict(data))
    data = index_lookup(data, tables, char_limit=params.char_limit,
                        num_parallel_calls=parallel_calls, has_labels=False)
    padded_shapes = get_padded_shapes(max_characters=params.char_limit, has_labels=False)
    data = data.padded_batch(
        batch_size=params.batch_size,
        padded_shapes=padded_shapes,
        drop_remainder=False
    )
    data = data.prefetch(buffer_size=params.max_prefetch)
    iterator = data.make_initializable_iterator()
    return data, iterator


def create_placeholders():
    """ Creates a dict of placeholder tensors for use in demo mode. """
    placeholders = {
        'context_tokens': tf.placeholder(shape=(None, None,), dtype=tf.string, name=' context_tokens'),
        'context_length': tf.placeholder(shape=(None,), dtype=tf.int32, name='context_length'),
        'query_tokens': tf.placeholder(shape=(None, None,), dtype=tf.string, name='query_tokens'),
        'query_length': tf.placeholder(shape=(None,), dtype=tf.int32, name='query_length')
    }

    return placeholders


def get_num_parallel_calls(params):
    """ Calculates the number of parallel calls we can make, if no number given returns the CPU count. """
    parallel_calls = params.parallel_calls
    if parallel_calls < 0:
        parallel_calls = os.cpu_count()
    return parallel_calls
