import tensorflow as tf
import numpy as np

bucket_by_sequence_length = tf.contrib.data.bucket_by_sequence_length
# useful link on pipelines: https://cs230-stanford.github.io/tensorflow-input-data.html


def create_placeholders(context_limit, query_limit, char_limit):
    ctxt_words = tf.placeholder(dtype=tf.int32, shape=(None, context_limit, ), name='context_words')
    ctxt_chars = tf.placeholder(dtype=tf.int32, shape=(None, context_limit, char_limit, ), name='context_chars')
    ctxt_len = tf.placeholder(dtype=tf.int32, shape=(None, ), name='context_length')
    query_words = tf.placeholder(dtype=tf.int32, shape=(None, query_limit,), name='query_words')
    query_chars = tf.placeholder(dtype=tf.int32, shape=(None, query_limit, char_limit,), name='query_chars')
    query_len = tf.placeholder(dtype=tf.int32, shape=(None, ), name='query_length')

    y_start = tf.placeholder(dtype=tf.int32, shape=(None, ), name='answer_start_index')
    y_end = tf.placeholder(dtype=tf.int32, shape=(None, ), name='answer_end_index')
    answer_id = tf.placeholder(dtype=tf.int32, shape=(None, ), name='answer_id')

    return ctxt_words, ctxt_chars, ctxt_len, query_words, query_chars, query_len, y_start, y_end, answer_id


def tf_record_pipeline(filenames, hparams, shuffle=True):
    int_feature = tf.FixedLenFeature([], tf.int32)

    features = {
        'context_words': tf.FixedLenSequenceFeature((hparams.context_limit, ), dtype=tf.int32),
        'context_chars': tf.FixedLenSequenceFeature((hparams.context_limit, hparams.char_limit), dtype=tf.int32),
        'context_length': int_feature,
        'query_words': tf.FixedLenSequenceFeature((hparams.query_limit, ), dtype=tf.int32),
        'query_chars': tf.FixedLenSequenceFeature((hparams.query_limit, hparams.char_limit), dtype=tf.int32),
        'answer_starts': int_feature,
        'answer_ends': int_feature,
        'answer_id': int_feature,
    }

    def parse(proto):
        return tf.parse_single_example(proto, features=features)

    dataset = tf.data.TFRecordDataset(filenames,
                                      buffer_size=hparams.tf_record_buffer_size,
                                      num_parallel_reads=hparams.parallel_calls)

    dataset = shuffle_repeat(dataset, shuffle, hparams.shuffle_buffer_size)

    dataset = dataset.map(parse, num_parallel_calls=hparams.parallel_calls)
    return dataset


def memory_pipeline(contexts, queries, context_mapping, hparams, shuffle=True):
    answer_ids = np.asarray((list(context_mapping.keys())), dtype=np.int32)
    np.random.shuffle(answer_ids)
    # Only store answer_ids for dynamic lookup.
    dataset = tf.data.Dataset.from_tensor_slices(answer_ids)
    dataset = shuffle_repeat(dataset, shuffle, hparams.shuffle_buffer_size)

    # As we have numerous repeated contexts each of great length, instead of storing on disk/in memory multiple
    # copies we dynamically retrieve it per batch, cuts down hard-drive usage by 2GB and RAM by 4GB with no
    # with no performance hit.
    def map_to_cache(answer_id):
        answer_key = str(answer_id)
        context_id = str(context_mapping[answer_key])
        context_words, context_chars, context_length = contexts[context_id]
        query_words, query_chars, query_length, answer_starts, answer_ends = queries[answer_key]
        return context_words, context_chars, context_length, query_words, query_chars, \
               query_length, answer_starts, answer_ends, answer_id

    dataset = dataset.map(lambda answer_id: tuple(tf.py_func(map_to_cache, [answer_id], [tf.int32] * 9)))
    return dataset


def length_fn(context_words, *args):
    context_words.set_shape([None])
    return tf.shape(context_words)[0]


def shuffle_repeat(dataset, shuffle, buffer_size):
    # Order of ops results in different shuffle per epoch.
    dataset = dataset.repeat()
    if shuffle:
        # Buffer size controls the random sampling, when buffer_size = length of data, shuffling is uniform.
        dataset = dataset.shuffle(buffer_size=buffer_size)
    return dataset


def create_buckets(hparams):
    # If no bucket ranges are explicitly defined, create using the bucket_size parameter
    if len(hparams.bucket_ranges) == 0:
        # Plus 1 as the bucket excludes the high number.
        return [i for i in range(0, hparams.context_limit + 1, hparams.bucket_size)]
    return hparams.bucket_ranges


def get_padded_shapes(hparams):
    return ([hparams.context_limit],  # context words
            [hparams.context_limit, hparams.char_limit],  # context chars
            [],  # context length
            [hparams.query_limit],  # query words
            [hparams.query_limit, hparams.char_limit],  # query chars
            [],  # query length
            [],  # answer_start
            [],  # answer_end
            [])  # answer_id


def create_pipeline(hparams, args, shuffle=True):
    if hparams.use_tf_record:
        record_paths = args
        dataset = tf_record_pipeline(record_paths, hparams, shuffle)
    else:
        contexts, queries, context_mapping = args
        dataset = memory_pipeline(contexts, queries, context_mapping, hparams, shuffle)
    # We either bucket (used in paper, faster train speed) or just form batches padded to max.
    # Note: py_func doesn't return output shapes therefore we zero pad to the limits on each batch and slice to
    # the batch max during training. @TODO revisit and see if this can be avoided in future tf versions.
    padded_shapes = get_padded_shapes(hparams)
    if hparams.bucket:
        buckets = create_buckets(hparams)
        dataset = dataset.apply(
            bucket_by_sequence_length(element_length_func=length_fn,
                                      bucket_batch_sizes=[hparams.batch_size] * (len(buckets) + 1),
                                      bucket_boundaries=buckets,
                                      padded_shapes=padded_shapes))
    else:
        dataset = dataset.padded_batch(
            batch_size=hparams.batch_size,
            padded_shapes=padded_shapes
        )

    # Prefetch in theory speeds up the pipeline by overlapping the batch generation and running previous batch.
    dataset = dataset.prefetch(buffer_size=hparams.max_prefetch)
    iterator = dataset.make_one_shot_iterator()
    return dataset, iterator
