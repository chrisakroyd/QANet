import tensorflow as tf
import numpy as np


def create_placeholders(context_limit, question_limit, char_limit):
    ctxt_words = tf.placeholder(dtype=tf.int32, shape=(None, context_limit, ), name='context_words')
    ctxt_chars = tf.placeholder(dtype=tf.int32, shape=(None, context_limit, char_limit, ), name='context_chars')
    ques_words = tf.placeholder(dtype=tf.int32, shape=(None, question_limit, ), name='question_words')
    ques_chars = tf.placeholder(dtype=tf.int32, shape=(None, question_limit, char_limit, ), name='question_chars')

    y_start = tf.placeholder(dtype=tf.int32, shape=(None, ), name='answer_start_index')
    y_end = tf.placeholder(dtype=tf.int32, shape=(None, ), name='answer_end_index')
    answer_id = tf.placeholder(dtype=tf.int32, shape=(None, ), name='answer_id')

    return ctxt_words, ctxt_chars, ques_words, ques_chars, y_start, y_end, answer_id

# useful link: https://cs230-stanford.github.io/tensorflow-input-data.html


def create_dataset(contexts, questions, context_mapping, hparams, shuffle=True, prefetch=2):
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
        context_words, context_chars = contexts[context_id]
        question_words, question_chars, answer_starts, answer_ends = questions[answer_key]
        return context_words, context_chars, question_words, question_chars, answer_starts, answer_ends, answer_id
    dataset = dataset.map(lambda answer_id: tuple(tf.py_func(map_to_cache, [answer_id], [tf.int32] * 7)))
    # We either bucket (used in paper, faster train speed) or just form batches padded to max.
    if hparams.bucket:
        raise NotImplementedError('Bucketing not yet implemented')
    else:
        dataset = dataset.padded_batch(
            batch_size=hparams.batch_size,
            padded_shapes=([hparams.context_limit],
                           [hparams.context_limit, hparams.char_limit],
                           [hparams.question_limit],
                           [hparams.question_limit, hparams.char_limit],
                           [],
                           [],
                           []),
        )

    # Prefetch in theory speeds up the pipeline by overlapping the batch generation and running previous batch.
    dataset = dataset.prefetch(prefetch)
    iterator = dataset.make_one_shot_iterator()
    return dataset, iterator
