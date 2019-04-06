import random
from operator import itemgetter
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tqdm import tqdm
from src import constants, preprocessing as prepro, config, util


def create_record(context, query):
    """ Creates a formatted tf.train Example for writing in a .tfrecord file. """
    encoded_context = [m.encode('utf-8') for m in context['tokens']]
    encoded_query = [m.encode('utf-8') for m in query['tokens']]

    context_tokens = tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_context))
    context_length = tf.train.Feature(int64_list=tf.train.Int64List(value=[context['length']]))
    query_tokens = tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_query))
    query_length = tf.train.Feature(int64_list=tf.train.Int64List(value=[query['length']]))
    answer_starts = tf.train.Feature(int64_list=tf.train.Int64List(value=[query['answer_starts']]))
    answer_ends = tf.train.Feature(int64_list=tf.train.Int64List(value=[query['answer_ends']]))
    answer_id = tf.train.Feature(int64_list=tf.train.Int64List(value=[query['answer_id']]))

    features = {
        'context_tokens': context_tokens,
        'context_length': context_length,
        'query_tokens': query_tokens,
        'query_length': query_length,
        'answer_starts': answer_starts,
        'answer_ends': answer_ends,
        'answer_id': answer_id,
    }

    if 'elmo' in context and 'elmo' in query:
        context_elmo = context['elmo'].reshape(-1)
        query_elmo = query['elmo'].reshape(-1)
        features['context_elmo'] = tf.train.Feature(float_list=tf.train.FloatList(value=context_elmo))
        features['query_elmo'] = tf.train.Feature(float_list=tf.train.FloatList(value=query_elmo))

    if 'is_impossible' in query:
        features['is_impossible'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[query['is_impossible']]))

    record = tf.train.Example(features=tf.train.Features(feature=features))

    return record


def dynamic_batch_size(offset, lengths, base_batch_size=32):
    """
        We have contexts up to 800 tokens long, this leads to OOM when we use too large of a static batch size, too low
        of a batch size and pre-processing takes up to an hour. This variable batch size allows us to use big
        batches early on for speed reasons and small later for memory reasons (Best of both worlds).

        TODO: Objectively fragile, only tested on a 1080ti and evaluated empirically, needs to be redone.

        Args:
            offset: Start index,
            lengths: 1-d array of lengths
            base_batch_size: Base batch size.
        Returns:
            Batch size value for a given context length.
    """
    batch_end_index = min(offset + base_batch_size, len(lengths))
    highest_length = lengths[batch_end_index - 1]  # We pre-sort by length and don't shuffle so this will be highest.
    batch_size = base_batch_size

    if highest_length > 400:
        batch_size = 4
    elif highest_length > 300:
        batch_size = 8
    elif highest_length > 250:
        batch_size = 16

    return batch_size


def extract_lists(rows, max_tokens=-1, skip_too_long=False, id_key='id'):
    """ Our data is in dicts but for convenience while batching we flatten into 3 lists to simplify . """
    all_tokens, all_lengths, all_ids, total = [], [], [], 0

    for row in rows:
        if not (row['length'] > max_tokens and skip_too_long):  # Skip rows > max_tokens
            all_tokens.append(row['tokens'])
            all_lengths.append(row['length'])
            all_ids.append(row[id_key])  # This should be universal and not specific to the queries. TODO: FIX.
            total += 1

    assert len(all_tokens) == len(all_lengths) == len(all_ids)

    return all_tokens, all_lengths, all_ids, total


def get_batch(tokens, lengths, ids, offset, total, batch_size=32):
    """ Gets a batch of size n starting at offset.  """
    batch_end_index = min(offset + batch_size, total)
    batch_tokens = tokens[offset:batch_end_index]
    batch_lengths = lengths[offset:batch_end_index]
    batch_ids = ids[offset:batch_end_index]
    assert len(batch_tokens) == len(batch_lengths) == len(batch_ids)
    batch_tokens = util.pad_to_max_length(batch_tokens, batch_lengths)  # Elmo requires constant sized arrays so we pad.
    return batch_tokens, batch_lengths, batch_ids


def shuffle_queries(queries):
    shuffled_queries = list(queries.values())
    random.shuffle(shuffled_queries)
    return shuffled_queries


def _write_with_contextual(path, contexts, queries, params, skip_too_long=True):
    """ Writes out a dataset as a .tfrecord file, pre-processing """
    # TODO: This whole function needs a major cleanup and a refactor.
    shuffled_queries = shuffle_queries(queries)

    with tf.python_io.TFRecordWriter(path) as writer, tf.Session(config=config.gpu_config()) as sess:
        elmo = hub.Module(constants.Urls.ELMO, trainable=False)
        sess.run(tf.global_variables_initializer())
        tokens_input = tf.placeholder(shape=(None, None,), dtype=tf.string)
        lengths_input = tf.placeholder(shape=(None,), dtype=tf.int32)
        embed_out = elmo(inputs={'tokens': tokens_input, 'sequence_len': lengths_input},
                         signature='tokens', as_dict=True)['elmo']

        # ElMo is expensive, we cut embed time by ensuring short sequences are batched together via sort by length
        rows = sorted(contexts.values(), key=itemgetter('length'))
        batch_size = 32
        tokens, lengths, ids, total = extract_lists(rows, params.max_tokens, skip_too_long, id_key='id')
        context_cache = {}

        print('embedding contexts...')
        with tqdm(total=total) as pbar:
            # We use a while loop rather than a for so we can have a dynamic batch size, this is for performance as
            # OOM occurs with large sequences and large batch size -> Only have small batch size for large sequences.
            i = 0
            while i < total:
                batch_size = dynamic_batch_size(i, lengths, batch_size)
                batch_tokens, batch_lengths, batch_ids = get_batch(tokens, lengths, ids, i, total, batch_size)

                elmo_out = sess.run(embed_out, feed_dict={
                    tokens_input: batch_tokens,
                    lengths_input: batch_lengths,
                })

                for row_id, embedding, length in zip(batch_ids, elmo_out, batch_lengths):
                    assert row_id in contexts
                    new_context = contexts[row_id].copy()
                    new_context['elmo'] = np.array(embedding[:length], dtype=np.float32)
                    context_cache[int(row_id)] = new_context
                    assert len(new_context['elmo']) == length

                i += len(batch_tokens)
                pbar.update(batch_size)

        # ElMo is expensive, we cut embed time by ensuring short sequences are batched together via sort by length
        id_key = 'answer_id'
        batch_size = 64
        tokens, lengths, ids, total = extract_lists(shuffled_queries, params.max_tokens, skip_too_long, id_key=id_key)

        print('embedding queries...')
        with tqdm(total=total) as pbar:
            i = 0
            while i < total:
                batch_tokens, batch_lengths, batch_ids = get_batch(tokens, lengths, ids, i, total, batch_size)

                elmo_out = sess.run(embed_out, feed_dict={
                    tokens_input: batch_tokens,
                    lengths_input: batch_lengths,
                })

                for row_id, embedding, length in zip(batch_ids, elmo_out, batch_lengths):
                    assert row_id in queries
                    query = queries[row_id]
                    context_id = int(query['context_id'])

                    if context_id in context_cache:
                        context = context_cache[context_id]
                        # We copy the dict as we don't want extremely large elmo arrays hanging around in memory.
                        query_copy = query.copy()
                        query_copy['elmo'] = np.array(embedding[:length], dtype=np.float32)
                        record = create_record(context, query_copy)
                        writer.write(record.SerializeToString())

                i += len(batch_tokens)
                pbar.update(batch_size)


def write_tf_record(path, contexts, queries, params, skip_too_long=True):
    """ Shuffles the queries and writes out the context + queries as a .tfrecord file optionally embedding with elmo.

        Args:
            path: Output path for the .tfrecord file.
            contexts: Dict mapping of context_id: words, spans + length (Context output from fit_and_extract)
            queries: Dict mapping of answer_id: words, answers, +start/end (Queries output from fit_and_extract)
            params: A dictionary of parameters.
            skip_too_long: Skip rows when either the query or context if their length is above max_tokens.
    """
    if params.use_contextual and params.fixed_contextual_embeddings:
        _write_with_contextual(path, contexts, queries, params, skip_too_long=skip_too_long)
    else:
        _write_tf_record(path, contexts, queries, params, skip_too_long=skip_too_long)


def _write_tf_record(path, contexts, queries, params, skip_too_long=True):
    """ Writes out context + queries for a dataset as a .tfrecord file. """
    shuffled_queries = shuffle_queries(queries)

    with tf.python_io.TFRecordWriter(path) as writer:
        for query in shuffled_queries:
            context_id = query['context_id']
            context = contexts[context_id]
            num_context_tokens = context['length']
            num_query_tokens = query['length']

            if (num_context_tokens > params.max_tokens or num_query_tokens > params.max_tokens) and skip_too_long:
                continue

            record = create_record(context, query)
            writer.write(record.SerializeToString())


def get_examples(contexts, queries, num_examples=1000):
    """ Gets a subsample of the contexts/queries.  """
    shuffled = list(queries.values())
    random.shuffle(shuffled)
    shuffled = shuffled[:num_examples]
    examples = [{'context': contexts[data['context_id']]['text'], 'query': data['text']} for data in shuffled]
    return examples


def convert_to_indices(text, tokens):
    """ Maps each token to its character offset within the text. """
    current = 0
    spans = []

    for token in tokens:
        next_toke_start = text.find(token, current)

        # We normalize all dash characters (e.g. en dash, em dash to -) which requires special handling when mapping.
        # TODO: @cakroyd look at this and improve implementation.
        if len(token) == 1 and prepro.is_dash(token):
            if prepro.is_dash(text[current]):
                current = current
            elif prepro.is_dash(text[current + 1]):
                current = current + 1
        else:
            current = next_toke_start

        if current == -1:
            print('Token {} cannot be found'.format(token))
            raise ValueError('Could not find token.')

        spans.append((current, current + len(token)))
        current += len(token)
    return spans
