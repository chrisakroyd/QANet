import tensorflow as tf


def create_placeholders(context_limit, question_limit, char_limit, size):
    ctxt_words = tf.placeholder(dtype=tf.int32, shape=(None, context_limit, ), name='context_words')
    ctxt_chars = tf.placeholder(dtype=tf.int32, shape=(None, context_limit, char_limit, ), name='context_chars')
    ques_words = tf.placeholder(dtype=tf.int32, shape=(None, question_limit, ), name='question_words')
    ques_chars = tf.placeholder(dtype=tf.int32, shape=(None, question_limit, char_limit, ), name='question_chars')

    y_start = tf.placeholder(dtype=tf.int32, shape=(None, ), name='answer_start_index')
    y_end = tf.placeholder(dtype=tf.int32, shape=(None, ), name='answer_end_index')
    answer_id = tf.placeholder(dtype=tf.int32, shape=(None, ), name='answer_id')

    return ctxt_words, ctxt_chars, ques_words, ques_chars, y_start, y_end, answer_id


def create_dataset(data, placeholders, batch_size, shuffle=True):
    context_words_input, context_chars_input, question_words_input, question_chars_input, \
    answer_starts_input, answer_ends_input, answer_ids_input = placeholders
    context_words, context_chars, question_words, question_chars, answer_starts, answer_ends, answer_ids = data
    # We create from tensor slices using placeholders so we can work in memory.
    data_set = tf.data.Dataset.from_tensor_slices(
        (context_words_input, context_chars_input, question_words_input, question_chars_input,
         answer_starts_input, answer_ends_input, answer_ids_input))
    # Shuffle and then batch either using max_examples or the total.
    # data_set = data_set.shuffle(len(context_words))
    if shuffle:
        data_set = data_set.shuffle(15000)
    data_set = data_set.repeat()
    data_set = data_set.batch(batch_size=batch_size)
    # Prefetch in theory speeds up the pipeline by overlapping the batch generation and running previous batch.
    data_set = data_set.prefetch(1)
    # Create the feed dict for initializing the iterator.
    feed_dict = {
        context_words_input: context_words,
        context_chars_input: context_chars,
        question_words_input: question_words,
        question_chars_input: question_chars,
        answer_starts_input: answer_starts,
        answer_ends_input: answer_ends,
        answer_ids_input: answer_ids
    }

    return data_set, feed_dict


def create_bucket_dataset(data, placeholders, batch_size, shuffle=True):
    context_words_input, context_chars_input, question_words_input, question_chars_input, \
    answer_starts_input, answer_ends_input, answer_ids_input = placeholders
    context_words, context_chars, question_words, question_chars, answer_starts, answer_ends, answer_ids = data
    # We create from tensor slices using placeholders so we can work in memory.
    data_set = tf.data.Dataset.from_tensor_slices(
        (context_words_input, context_chars_input, question_words_input, question_chars_input,
         answer_starts_input, answer_ends_input, answer_ids_input))
    # Shuffle and then batch either using max_examples or the total.
    # data_set = data_set.shuffle(len(context_words))
    if shuffle:
        data_set = data_set.shuffle(15000)
    data_set = data_set.repeat()
    data_set = data_set.batch(batch_size=batch_size)
    # Prefetch in theory speeds up the pipeline by overlapping the batch generation and running previous batch.
    data_set = data_set.prefetch(1)
    # Create the feed dict for initializing the iterator.
    feed_dict = {
        context_words_input: context_words,
        context_chars_input: context_chars,
        question_words_input: question_words,
        question_chars_input: question_chars,
        answer_starts_input: answer_starts,
        answer_ends_input: answer_ends,
        answer_ids_input: answer_ids
    }

    return data_set, feed_dict
