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


def length_fn(context_words, *args):
    context_words.set_shape([None])
    return tf.shape(context_words)[0]


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


def create_dataset(contexts, queries, context_mapping, hparams, shuffle=True):
    # Extract an array of all answer_ids.
    answer_ids = np.asarray(list(context_mapping.keys()), dtype=np.int32)
    # Only store answer_ids for dynamic lookup.
    dataset = tf.data.Dataset.from_tensor_slices(answer_ids)
    # Order of ops results in different shuffle per epoch.
    dataset = dataset.repeat()
    if shuffle:
        # Buffer size controls the random sampling, when buffer_size = length of data, shuffling is uniform.
        dataset = dataset.shuffle(buffer_size=hparams.shuffle_buffer_size)

    # As we have numerous repeated contexts each of great length, instead of storing on disk/in memory multiple
    # copies we dynamically retrieve it per batch, cuts down hard-drive usage by 2GB and RAM by 4GB with no
    # with no performance hit.
    def map_to_cache(answer_id):
        answer_key = str(answer_id)
        context_id = str(context_mapping[answer_key])
        context_words, context_chars, context_length = contexts[context_id]
        query_words, query_chars, query_length, answer_starts, answer_ends = queries[answer_key]
        return context_words, context_chars, context_length, query_words, query_chars,\
               query_length, answer_starts, answer_ends, answer_id

    dataset = dataset.map(lambda answer_id: tuple(tf.py_func(map_to_cache, [answer_id], [tf.int32] * 9)))
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
