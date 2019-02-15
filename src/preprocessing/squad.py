import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from src import preprocessing as prepro, util

keys_to_remove = ['tokens', 'length', 'query', 'elmo']


def convert_idx(text, tokens):
    """ Maps each token to its character offset within the text. """
    current = 0
    spans = []

    for token in tokens:
        next_toke_start = text.find(token, current)

        # We normalize all dash characters (e.g. en dash, em dash to -) which requires special handling when mapping.
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


def read_squad_examples(path):
    """
        Loads squad examples and flattens the data into two dicts, one for contexts + one for questions.
        Args:
            path: A filepath to a squad v1 or v2 .json file.
        Returns:
            Two dicts of unprocessed data, one containing contexts and one containing question/answer pairs.
    """
    squad_data = util.load_json(path)

    contexts, queries = {}, {}
    context_id, answer_id = 1, 1

    for data in squad_data['data']:
        for paragraph in data['paragraphs']:
            context_orig = paragraph['context'].strip()

            for qa in paragraph['qas']:
                orig_ques = qa['question'].strip()
                orig_answers = qa['answers']

                query_dict = {
                    'id': qa['id'],
                    'answer_id': answer_id,
                    'context_id': context_id,
                    'orig_text': orig_ques,
                    'orig_answers': orig_answers,
                }

                if 'is_impossible' in qa:
                    queries['is_impossible'] = qa['is_impossible']

                assert answer_id not in queries
                queries[answer_id] = query_dict
                answer_id += 1

            assert context_id not in contexts

            contexts[context_id] = {
                'id': context_id,
                'orig_text': context_orig,
            }

            context_id += 1

    return contexts, queries


def fit_and_extract(path, tokenizer, skip_on_errors=True):
    """ Loads squad file, flattens the data structure, fits the tokenizer + preprocesses the data.
        Args:
            path: A filepath to a squad v1 or v2 .json file.
            tokenizer: A Tokenizer instance.
            skip_on_errors: Whether or not to include answers which can't be found in the data.
        Returns:
            Two dicts of processed data, one containing contexts and one containing question/answer pairs.
    """
    contexts, queries = read_squad_examples(path)
    processed_queries = {}

    for key, context in contexts.items():
        orig_text = context['orig_text'].strip()
        clean_text = prepro.clean(orig_text)
        tokens = tokenizer.fit_on_texts(clean_text)[-1]
        token_orig_map = convert_idx(orig_text, tokens)

        context.update({
            'text': clean_text,
            'tokens': tokens,
            'length': len(tokens),
            'token_to_orig_map': token_orig_map,
        })

    for key, query in tqdm(queries.items()):
        context = contexts[query['context_id']]
        orig_text = query['orig_text'].strip()
        clean_text = prepro.clean(orig_text)

        query_tokens = tokenizer.fit_on_texts(clean_text)[-1]
        answer_starts, answer_ends, answer_texts, orig_answer_texts = [], [], [], []

        for answer in query['orig_answers']:
            answer_orig = answer['text'].strip()
            answer_text = prepro.clean(answer_orig)
            answer_start = answer['answer_start']
            answer_end = answer_start + len(answer_orig)

            # Soft warning, not necessarily a failure as the answer may not be exactly present or due to unicode errors
            # e.g. answer_text = Destiny Fulfilled. when only present in text as Destiny Fulfilled ...
            if context['text'].find(answer_text) == -1:
                print('Cannot find answer in text for question id {}'.format(query['id']))

            answer_span = []

            for i, span in enumerate(context['token_to_orig_map']):
                if not (answer_end <= span[0] or answer_start >= span[1]):
                    answer_span.append(i)

            # Usually as a result of mis-labelling in the dataset, we skip for train but include in dev/test modes.
            if len(answer_span) == 0:
                if skip_on_errors:
                    print('Cannot find span in text for question id {}'.format(query['id']))
                    continue
                else:
                    answer_span.append((0, 0, ))  # If we don't skip simply use this placeholder pointer.

            assert answer_span[-1] >= answer_span[0]

            orig_answer_texts.append(answer_orig)
            answer_texts.append(answer_text)
            answer_starts.append(answer_span[0])
            answer_ends.append(answer_span[-1])

        assert len(answer_starts) == len(answer_ends) == len(answer_texts)

        if len(answer_starts) == 0 or len(answer_ends) == 0:
            continue

        query.update({
            'text': orig_text,
            'tokens': query_tokens,
            'length': len(query_tokens),
            'orig_answer_texts': orig_answer_texts,
            'answers': answer_texts,
            'answer_starts': answer_starts[-1],
            'answer_ends': answer_ends[-1],
        })

        processed_queries[key] = query

    total = len(queries)
    processed = len(processed_queries)

    print('Total: {}'.format(total))
    print('Total Answers: {}'.format(processed))
    print('Total Skipped: {}'.format(total - processed))

    return contexts, processed_queries, tokenizer


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


def write_as_tf_record(path, contexts, queries, params, skip_too_long=True):
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


def process(params):
    directories = util.get_directories(params)
    util.make_dirs(directories)
    # Path to squad
    train_raw_path, dev_raw_path = util.raw_data_paths(params)
    train_record_path, dev_record_path, test_record_path = util.tf_record_paths(params)
    examples_path = util.examples_path(params)
    # Paths for dumping our processed data.
    train_contexts_path, train_answers_path, dev_contexts_path, dev_answers_path, \
    test_contexts_path, test_answers_path = util.processed_data_paths(params)
    # Get paths for saving embedding related info.
    word_index_path, trainable_index_path, char_index_path = util.index_paths(params)
    word_embeddings_path, trainable_embeddings_path, char_embeddings_path = util.embedding_paths(params)

    # Read the embedding index and create a vocab of words with embeddings.
    print('Loading Embeddings, this may take some time...')
    embedding_index = util.read_embeddings_file(params.embeddings_path)
    vocab = set([e for e, _ in embedding_index.items()])

    tokenizer = util.Tokenizer(max_words=params.max_words + 1,
                               max_chars=params.max_chars + 1,
                               vocab=vocab,
                               lower=False,
                               oov_token=params.oov_token,
                               min_word_occurrence=params.min_word_occur,
                               min_char_occurrence=params.min_char_occur,
                               trainable_words=params.trainable_words,
                               filters=None)

    print('Processing...')
    train_contexts, train_answers, tokenizer = fit_and_extract(train_raw_path, tokenizer)
    dev_contexts, dev_answers, tokenizer = fit_and_extract(dev_raw_path, tokenizer, skip_on_errors=False)
    tokenizer.init()
    word_index = tokenizer.word_index
    char_index = tokenizer.char_index
    trainable_index = util.index_from_list(params.trainable_words)

    embedding_matrix = util.load_embedding_file(path=params.embeddings_path,
                                                word_index=word_index,
                                                embedding_dimensions=params.embed_dim,
                                                trainable_embeddings=params.trainable_words,
                                                embedding_index=embedding_index)

    trainable_matrix = util.generate_matrix(index=trainable_index, embedding_dimensions=params.embed_dim)
    char_matrix = util.generate_matrix(index=char_index, embedding_dimensions=params.char_dim)

    print('Saving to TF Records...')
    write_as_tf_record(train_record_path, train_contexts, train_answers, params)
    write_as_tf_record(dev_record_path, dev_contexts, dev_answers, params)
    write_as_tf_record(test_record_path, dev_contexts, dev_answers, params, skip_too_long=False)
    examples = get_examples(train_contexts, train_answers)

    train_contexts = util.remove_keys(train_contexts, keys_to_remove)
    dev_contexts = util.remove_keys(dev_contexts, keys_to_remove)
    train_answers = util.remove_keys(train_answers, keys_to_remove)
    dev_answers = util.remove_keys(dev_answers, keys_to_remove)

    # Save the generated data
    util.save_json(train_contexts_path, train_contexts)
    util.save_json(train_answers_path, train_answers)
    util.save_json(dev_contexts_path, dev_contexts)
    util.save_json(dev_answers_path, dev_answers)
    util.save_json(test_contexts_path, dev_contexts)
    util.save_json(test_answers_path, dev_answers)
    util.save_json(examples_path, examples)
    # Save the word index mapping of word:index for both the pre-trained and trainable embeddings.
    util.save_json(word_index_path, word_index)
    util.save_json(char_index_path, char_index)
    util.save_json(trainable_index_path, trainable_index)
    # Save embeddings
    np.save(trainable_embeddings_path, trainable_matrix)
    np.save(word_embeddings_path, embedding_matrix)
    np.save(char_embeddings_path, char_matrix)
