import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from src import preprocessing as prepro, util

context_keys_to_remove = ['context_tokens']
answer_keys_to_remove = ['query_tokens', 'query']


def convert_idx(text, tokens):
    current = 0
    spans = []

    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print('Token {} cannot be found'.format(token))
            raise ValueError('Could not find token.')
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def fit_and_extract(data_set, tokenizer, params):
    """ Extracts context + query tokens and flattens squad data structure.
        Args:
            data_set: Either train/dev squad dict.
            tokenizer: A Tokenizer instance.
            params: A dictionary of parameters.
        Returns:
            Contexts, Queries and the tokenizer.
    """
    contexts, queries = {}, {}
    context_id, answer_id, total = 1, 1, 0

    for data in tqdm(data_set['data']):
        for question_answer in data['paragraphs']:
            context_clean = prepro.clean(question_answer['context'])
            # Fit the tokenizer on the cleaned version of the context.
            context_tokens = tokenizer.fit_on_texts(context_clean)[-1]
            if len(context_tokens) > params.max_tokens:
                continue
            spans = convert_idx(context_clean, context_tokens)

            for qa in question_answer['qas']:
                query_clean = prepro.clean(qa['question'])
                query_tokens = tokenizer.fit_on_texts(query_clean)[-1]
                total += 1
                answer_starts, answer_ends, answer_texts = [], [], []

                for answer in qa['answers']:
                    answer_text = prepro.clean(answer['text'])
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer_text)

                    assert context_clean.find(answer_text) >= 0

                    answer_texts.append(answer_text)
                    answer_span = []

                    for i, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(i)

                    assert answer_span[-1] >= answer_span[0]
                    assert len(answer_span) > 0

                    answer_starts.append(answer_span[0])
                    answer_ends.append(answer_span[-1])

                assert len(answer_starts) == len(answer_ends) == len(answer_texts)
                assert answer_id not in queries

                queries[answer_id] = {
                    'id': qa['id'],
                    'answer_id': answer_id,
                    'context_id': context_id,
                    'query': query_clean,
                    'query_tokens': query_tokens,
                    'query_length': len(query_tokens),
                    'answers': answer_texts,
                    'answer_starts': answer_starts[-1],
                    'answer_ends': answer_ends[-1],
                }

                answer_id += 1

            assert context_id not in contexts

            contexts[context_id] = {
                'id': context_id,
                'context': context_clean,
                'context_tokens': context_tokens,
                'context_length': len(context_tokens),
                'word_spans': spans,
            }

            context_id += 1

    assert len(queries) == total

    return contexts, queries, tokenizer


def write_as_tf_record(path, contexts, queries):
    """ Shuffles the queries and writes out the context + queries as a .tfrecord file.

        Args:
            path: Output path for the .tfrecord file.
            contexts: Dict mapping of context_id: words, spans + length (Context output from fit_and_extract)
            queries: Dict mapping of answer_id: words, answers, +start/end (Queries output from fit_and_extract)
    """
    records = []
    shuffled = list(queries.values())
    random.shuffle(shuffled)
    for data in shuffled:
        context = contexts[data['context_id']]
        encoded_context = [m.encode('utf-8') for m in context['context_tokens']]
        encoded_query = [m.encode('utf-8') for m in data['query_tokens']]

        context_tokens = tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_context))
        context_length = tf.train.Feature(int64_list=tf.train.Int64List(value=[context['context_length']]))
        query_tokens = tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_query))
        query_length = tf.train.Feature(int64_list=tf.train.Int64List(value=[data['query_length']]))
        answer_starts = tf.train.Feature(int64_list=tf.train.Int64List(value=[data['answer_starts']]))
        answer_ends = tf.train.Feature(int64_list=tf.train.Int64List(value=[data['answer_ends']]))
        answer_id = tf.train.Feature(int64_list=tf.train.Int64List(value=[data['answer_id']]))

        record = tf.train.Example(features=tf.train.Features(feature={
            'context_tokens': context_tokens,
            'context_length': context_length,
            'query_tokens': query_tokens,
            'query_length': query_length,
            'answer_starts': answer_starts,
            'answer_ends': answer_ends,
            'answer_id': answer_id,
        }))

        records.append(record)

    with tf.python_io.TFRecordWriter(path) as writer:
        for record in records:
            writer.write(record.SerializeToString())


def get_examples(contexts, queries, num_examples=1000):
    shuffled = list(queries.values())
    random.shuffle(shuffled)
    shuffled = shuffled[:num_examples]
    examples = [{'context': contexts[data['context_id']]['context'], 'query': data['query']} for data in shuffled]
    return examples


def process(params):
    # Create prerequisite directories.
    directories = util.get_directories(params)
    util.make_dirs(directories)
    # Path to squad
    train_path, dev_path = util.raw_data_paths(params)
    # path to save tf_records and a random sample of data.
    train_record_path = util.tf_record_paths(params, training=True)
    dev_record_path = util.tf_record_paths(params, training=False)
    examples_path = util.examples_path(params)
    # Paths for dumping our processed data.
    train_contexts_path, train_answers_path, dev_contexts_path, dev_answers_path = util.processed_data_paths(params)
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
                               char_limit=params.char_limit,
                               min_word_occurrence=params.min_word_occur,
                               min_char_occurrence=params.min_char_occur,
                               trainable_words=params.trainable_words,
                               filters=None)

    train = util.load_json(train_path)
    dev = util.load_json(dev_path)

    print('Processing...')
    train_contexts, train_answers, tokenizer = fit_and_extract(train, tokenizer, params)
    dev_contexts, dev_answers, tokenizer = fit_and_extract(dev, tokenizer, params)
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
    write_as_tf_record(train_record_path, train_contexts, train_answers)
    write_as_tf_record(dev_record_path, dev_contexts, dev_answers)
    examples = get_examples(train_contexts, train_answers)

    train_contexts = util.remove_keys(train_contexts, context_keys_to_remove)
    dev_contexts = util.remove_keys(dev_contexts, context_keys_to_remove)
    train_answers = util.remove_keys(train_answers, answer_keys_to_remove)
    dev_answers = util.remove_keys(dev_answers, answer_keys_to_remove)

    # Save the generated data
    util.save_json(train_contexts_path, train_contexts)
    util.save_json(train_answers_path, train_answers)
    util.save_json(dev_contexts_path, dev_contexts)
    util.save_json(dev_answers_path, dev_answers)
    # Save a random sample of the data.
    util.save_json(examples_path, examples)
    # Save the word index mapping of word:index for both the pre-trained and trainable embeddings.
    util.save_json(word_index_path, word_index)
    util.save_json(char_index_path, char_index)
    util.save_json(trainable_index_path, trainable_index)
    # Save the trainable embeddings matrix.
    np.save(trainable_embeddings_path, trainable_matrix)
    # Save the full embeddings matrix
    np.save(word_embeddings_path, embedding_matrix)
    np.save(char_embeddings_path, char_matrix)
