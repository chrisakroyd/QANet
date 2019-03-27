import tensorflow as tf
import random
from src import preprocessing as prepro


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

    if 'is_impossible' in query:
        features['is_impossible'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[query['is_impossible']]))

    record = tf.train.Example(features=tf.train.Features(feature=features))

    return record


def write_tf_record(path, contexts, queries, params, skip_too_long=True):
    """ Shuffles the queries and writes out the context + queries as a .tfrecord file.

        Args:
            path: Output path for the .tfrecord file.
            contexts: Dict mapping of context_id: words, spans + length (Context output from fit_and_extract)
            queries: Dict mapping of answer_id: words, answers, +start/end (Queries output from fit_and_extract)
            params: A dictionary of parameters.
            skip_too_long: Skip rows when either the query or context if their length is above max_tokens.
    """
    shuffled = list(queries.values())
    random.shuffle(shuffled)

    with tf.python_io.TFRecordWriter(path) as writer:
        for data in shuffled:
            context_id = data['context_id']
            context = contexts[context_id]
            num_context_tokens = context['length']
            num_query_tokens = data['length']

            if (num_context_tokens > params.max_tokens or num_query_tokens > params.max_tokens) and skip_too_long:
                continue

            record = create_record(context, data)
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
