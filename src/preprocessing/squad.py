import tensorflow as tf
import numpy as np
from tqdm import tqdm
from src import preprocessing as prepro, util


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


def fit_and_extract(data_set, tokenizer, hparams):
    contexts, queries = {}, {}
    context_id, answer_id, skipped, total = 1, 1, 0, 0

    for data in tqdm(data_set['data']):
        for question_answer in data['paragraphs']:
            context_clean = prepro.clean(question_answer['context'])
            # Fit the tokenizer on the cleaned version of the context.
            context_tokens = tokenizer.fit_on_texts(context_clean)[-1]
            if len(context_tokens) > hparams.context_limit:
                continue
            spans = convert_idx(context_clean, context_tokens)

            for qa in question_answer['qas']:
                query_clean = prepro.clean(qa['question'])
                query_tokens = tokenizer.fit_on_texts(query_clean)[-1]
                total += 1

                if len(query_tokens) > hparams.query_limit:
                    skipped += 1
                    continue
                answer_starts, answer_ends, answer_texts = [], [], []

                for answer in qa['answers']:
                    answer_text = prepro.clean(answer['text'])
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer_text)
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

    assert len(queries) == (total - skipped)

    return contexts, queries, tokenizer


def write_as_tf_record(path, contexts, queries):
    records = []
    for data in queries.values():
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


def remove_unneeded_keys(data, keys=[]):
    for _, value in data.items():
        for key in keys:
            value.pop(key, None)
    return data


def process(hparams):
    train_path, dev_path = util.raw_data_paths(hparams)
    directories = util.get_directories(hparams)
    util.make_dirs(directories)
    # Paths for dumping our processed data.
    train_contexts_path, train_answers_path, dev_contexts_path, dev_answers_path = util.processed_data_paths(hparams)
    # Get paths for saving embedding related info.
    word_index_path, trainable_index_path, char_index_path = util.index_paths(hparams)
    word_embeddings_path, trainable_embeddings_path, char_embeddings_path = util.embedding_paths(hparams)
    # Read the embedding index and create a vocab of words with embeddings.
    print('Loading Embeddings, this may take some time...')
    embedding_index = util.read_embeddings_file(hparams.embeddings_path)
    vocab = util.create_vocab(embedding_index)

    tokenizer = util.Tokenizer(max_words=hparams.max_words + 1,
                               max_chars=hparams.max_chars + 1,
                               vocab=vocab,
                               lower=False,
                               oov_token=hparams.oov_token,
                               char_limit=hparams.char_limit,
                               min_word_occurrence=hparams.min_word_occur,
                               min_char_occurrence=hparams.min_char_occur,
                               trainable_words=hparams.trainable_words,
                               filters=None)

    train = util.load_json(train_path)
    dev = util.load_json(dev_path)

    print('Processing...')
    train_contexts, train_answers, tokenizer = fit_and_extract(train, tokenizer, hparams)
    dev_contexts, dev_answers, tokenizer = fit_and_extract(dev, tokenizer, hparams)
    tokenizer.init()
    word_index = tokenizer.word_index
    char_index = tokenizer.char_index
    trainable_index = util.index_from_list(hparams.trainable_words)

    embedding_matrix = util.load_embedding(path=hparams.embeddings_path,
                                           word_index=word_index,
                                           embedding_dimensions=hparams.embed_dim,
                                           trainable_embeddings=hparams.trainable_words,
                                           embedding_index=embedding_index)

    trainable_matrix = util.generate_matrix(index=trainable_index, embedding_dimensions=hparams.embed_dim)
    char_matrix = util.generate_matrix(index=char_index, embedding_dimensions=hparams.char_dim)

    print('Saving to TF Records...')
    train_path = util.tf_record_paths(hparams, train=True)
    dev_path = util.tf_record_paths(hparams, train=False)
    write_as_tf_record(train_path, train_contexts, train_answers)
    write_as_tf_record(dev_path, dev_contexts, dev_answers)

    train_contexts = remove_unneeded_keys(train_contexts, ['context_tokens', 'context_length'])
    dev_contexts = remove_unneeded_keys(dev_contexts, ['context_tokens', 'context_length'])
    train_answers = remove_unneeded_keys(train_answers, ['query_tokens', 'query_length'])
    dev_answers = remove_unneeded_keys(dev_answers, ['query_tokens', 'query_length'])

    # Save the generated data
    util.save_json(train_contexts_path, train_contexts)
    util.save_json(train_answers_path, train_answers)
    util.save_json(dev_contexts_path, dev_contexts)
    util.save_json(dev_answers_path, dev_answers)
    # Save the word index mapping of word:index for both the pre-trained and trainable embeddings.
    util.save_json(word_index_path, word_index)
    util.save_json(char_index_path, char_index)
    util.save_json(trainable_index_path, trainable_index)
    # Save the trainable embeddings matrix.
    np.save(trainable_embeddings_path, trainable_matrix)
    # Save the full embeddings matrix
    np.save(word_embeddings_path, embedding_matrix)
    np.save(char_embeddings_path, char_matrix)
