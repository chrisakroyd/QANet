import tensorflow as tf

bucket_by_sequence_length = tf.contrib.data.bucket_by_sequence_length
# useful link on pipelines: https://cs230-stanford.github.io/tensorflow-input-data.html


def tf_record_pipeline(filenames, hparams):
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
                                      buffer_size=hparams.tf_record_buffer_size,
                                      num_parallel_reads=hparams.parallel_calls)

    dataset = dataset.map(parse, num_parallel_calls=hparams.parallel_calls)
    return dataset


def index_lookup(dataset, word_table, char_table, char_limit=16, num_parallel_calls=4):
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


def length_fn(fields):
    return tf.cast(fields['context_length'], dtype=tf.int32)


def create_buckets(hparams):
    # If no bucket ranges are explicitly defined, create using the bucket_size parameter
    if len(hparams.bucket_ranges) == 0:
        # Plus 1 as the bucket excludes the high number.
        return [i for i in range(0, hparams.context_limit + 1, hparams.bucket_size)]
    return hparams.bucket_ranges


def get_padded_shapes(hparams):
    return {'context_words': [hparams.context_limit],  # context words
            'context_chars': [hparams.context_limit, hparams.char_limit],  # context chars
            'context_length': [],  # context length
            'query_words': [hparams.query_limit],  # query words
            'query_chars': [hparams.query_limit, hparams.char_limit],  # query chars
            'query_length': [],  # query length
            'answer_starts': [],  # answer_start
            'answer_ends': [],  # answer_end
            'answer_id': []}  # answer_id


def create_lookup_tables(word_vocab, char_vocab):
    # default value is highest in the vocab as this is the OOV embedding, we generate non-zero indexed therefore -1.
    word_table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(word_vocab, dtype=tf.string),
                                                           default_value=len(word_vocab) - 1)
    char_table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(char_vocab, dtype=tf.string),
                                                           default_value=len(char_vocab) - 1)
    return word_table, char_table


def create_pipeline(hparams, word_table, char_table, record_paths, train=True):
    dataset = tf_record_pipeline(record_paths, hparams)
    dataset = dataset.cache().repeat()
    if train:
        dataset = dataset.shuffle(buffer_size=hparams.shuffle_buffer_size)
    # Perform word -> index mapping.
    dataset = index_lookup(dataset, word_table, char_table, char_limit=hparams.char_limit,
                           num_parallel_calls=hparams.parallel_calls)
    # We either bucket (used in paper, faster train speed) or just form batches padded to max.
    # Note: py_func doesn't return output shapes therefore we zero pad to the limits on each batch and slice to
    # the batch max during training. @TODO revisit and see if this can be avoided.
    padded_shapes = get_padded_shapes(hparams)
    if hparams.bucket and train:
        buckets = create_buckets(hparams)
        dataset = dataset.apply(
            bucket_by_sequence_length(element_length_func=length_fn,
                                      bucket_batch_sizes=[hparams.batch_size] * (len(buckets) + 1),
                                      bucket_boundaries=buckets,
                                      padded_shapes=padded_shapes))
    else:
        dataset = dataset.padded_batch(
            batch_size=hparams.batch_size,
            padded_shapes=padded_shapes,
            drop_remainder=train
        )

    dataset = dataset.prefetch(buffer_size=hparams.max_prefetch)
    iterator = dataset.make_initializable_iterator()
    return dataset, iterator
