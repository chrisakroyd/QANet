import numpy as np
from tqdm import tqdm
from src import preprocessing as prepro, util, tokenizer as toke

keys_to_remove = ['tokens', 'length', 'query', 'elmo', 'orig_tokens']


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

    for key, context in tqdm(contexts.items()):
        orig_text = context['orig_text'].strip()
        clean_text = prepro.clean(orig_text)
        tokens, modified_tokens = tokenizer.fit_on_texts(clean_text, error_correct=True)[-1]
        token_orig_map = prepro.convert_to_indices(orig_text, tokens)

        context.update({
            'text': clean_text,
            'orig_tokens': tokens,
            'tokens': modified_tokens,
            'length': len(modified_tokens),
            'token_to_orig_map': token_orig_map,
        })

    for key, query in tqdm(queries.items()):
        context = contexts[query['context_id']]
        orig_text = query['orig_text'].strip()
        clean_text = prepro.clean(orig_text)

        tokens, modified_tokens = tokenizer.fit_on_texts(clean_text, error_correct=True)[-1]
        answer_starts, answer_ends, answer_texts, orig_answer_texts = [], [], [], []

        for answer in query['orig_answers']:
            answer_orig = answer['text'].strip()
            answer_text = prepro.clean(answer_orig)
            answer_start = answer['answer_start']
            answer_end = answer_start + len(answer_orig)

            # Soft warning, not necessarily a failure as the answer may not be exactly present or due to unicode errors
            # e.g. answer_text = Destiny Fulfilled. when only present in text as Destiny Fulfilled ...
            if context['text'].find(answer_text) == -1:
                print('Cannot find exact answer in text for question id {}'.format(query['id']))

            answer_span = []

            for i, span in enumerate(context['token_to_orig_map']):
                if not (answer_end <= span[0] or answer_start >= span[1]):
                    answer_span.append(i)

            # Usually as a result of mis-labelling in the dataset, we skip for train but include in dev/test modes.
            if len(answer_span) == 0:
                if skip_on_errors:
                    print('Cannot find answer span for question id {}'.format(query['id']))
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
            'tokens': modified_tokens,
            'length': len(modified_tokens),
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

    tokenizer = toke.Tokenizer(max_words=params.max_words + 1,
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

    trainable_matrix = util.generate_matrix(index=trainable_index, embedding_dimensions=params.embed_dim, scale=0.1)
    char_matrix = util.generate_matrix(index=char_index, embedding_dimensions=params.char_dim, scale=0.1)

    print('Saving to TF Records...')

    if params.use_contextual and params.fixed_contextual_embeddings:
        record_writer = prepro.ContextualEmbeddingWriter(params.max_tokens, contextual_model=params.contextual_model)
    else:
        record_writer = prepro.RecordWriter(params.max_tokens)

    record_writer.write(train_record_path, train_contexts, train_answers)
    record_writer.write(dev_record_path, dev_contexts, dev_answers)
    record_writer.write(test_record_path, dev_contexts, dev_answers, skip_too_long=False)

    examples = prepro.get_examples(train_contexts, train_answers)

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
