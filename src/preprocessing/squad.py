import numpy as np
from tqdm import tqdm
from .text import clean
from src.util import raw_data_paths, processed_data_paths, embedding_paths, index_paths
from src.util import Tokenizer, generate_matrix, load_embedding, save_json, load_json, index_from_list,\
    read_embeddings_file, create_vocab


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
    context_id, answer_id = 1, 1

    for data in tqdm(data_set['data']):
        for question_answer in data['paragraphs']:
            context_clean = clean(question_answer['context'])
            # Fit the tokenizer on the cleaned version of the context.
            context_tokens = tokenizer.fit_on_texts(context_clean)[-1]
            if len(context_tokens) > hparams.context_limit:
                continue
            spans = convert_idx(context_clean, context_tokens)

            for qa in question_answer['qas']:
                query_clean = clean(qa['question'])
                query_tokens = tokenizer.fit_on_texts(query_clean)[-1]
                if len(query_tokens) > hparams.query_limit:
                    continue
                answer_starts, answer_ends, answer_texts = [], [], []

                for answer in qa['answers']:
                    answer_text = clean(answer['text'])
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer_text)
                    answer_texts.append(answer_text)
                    answer_span = []

                    for i, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(i)

                    answer_starts.append(answer_span[0])
                    answer_ends.append(answer_span[-1])

                if (answer_ends[-1] - answer_starts[-1]) > hparams.answer_limit:
                    continue

                queries[answer_id] = {
                    'id': qa['id'],
                    'answer_id': answer_id,
                    'context_id': context_id,
                    'query': query_clean,
                    'query_tokens': query_tokens,
                    'answers': answer_texts,
                    'answer_starts': answer_starts[-1],
                    'answer_ends': answer_ends[-1],
                }

                answer_id += 1

            contexts[context_id] = {
                'id': context_id,
                'context': context_clean,
                'context_tokens': context_tokens,
                'word_spans': spans,
            }

            context_id += 1

    return contexts, queries, tokenizer


def text_to_sequence(data, text_key, save_key, tokenizer, max_words):
    for key, value in tqdm(data.items()):
        words, chars, _ = tokenizer.tokens_to_sequences(value[text_key],
                                                        seq_length=max_words,
                                                        pad=False)
        data[key]['{}_words'.format(save_key)] = words[-1]
        data[key]['{}_chars'.format(save_key)] = chars[-1]
    return data


def pre_process(contexts, question_answers, tokenizer, hparams):
    contexts = text_to_sequence(contexts, 'context_tokens', 'context', tokenizer, hparams.context_limit)
    question_answers = text_to_sequence(question_answers, 'query_tokens', 'query', tokenizer,
                                        hparams.query_limit)
    return contexts, question_answers


def process(hparams):
    train_path, dev_path = raw_data_paths(hparams)
    # Paths for dumping our processed data.
    train_contexts_path, train_answers_path, dev_contexts_path, dev_answers_path = processed_data_paths(hparams)
    # Get paths for saving embedding related info.
    word_index_path, trainable_index_path, char_index_path = index_paths(hparams)
    word_embeddings_path, trainable_embeddings_path, char_embeddings_path = embedding_paths(hparams)
    # Read the embedding index and create a vocab of words with embeddings.
    print('Loading Embeddings, this may take some time...')
    embedding_index = read_embeddings_file(hparams.embeddings_path)
    vocab = create_vocab(embedding_index)

    tokenizer = Tokenizer(max_words=hparams.max_words + 1,
                          max_chars=hparams.max_chars + 1,
                          vocab=vocab,
                          lower=False,
                          oov_token=hparams.oov_token,
                          char_limit=hparams.char_limit,
                          min_word_occurrence=hparams.min_word_occur,
                          min_char_occurrence=hparams.min_char_occur,
                          trainable_words=hparams.trainable_words,
                          filters=None)

    train = load_json(train_path)
    dev = load_json(dev_path)

    print('Fitting...')
    train_contexts, train_question_answers, tokenizer = fit_and_extract(train, tokenizer, hparams)
    dev_contexts, dev_question_answers, tokenizer = fit_and_extract(dev, tokenizer, hparams)

    print('Processing...')
    train_contexts, train_question_answers = pre_process(train_contexts, train_question_answers, tokenizer, hparams)
    dev_contexts, dev_question_answers = pre_process(dev_contexts, dev_question_answers, tokenizer, hparams)

    word_index = tokenizer.word_index
    char_index = tokenizer.char_index
    trainable_word_index = index_from_list(hparams.trainable_words)

    embedding_matrix, trainable_matrix = load_embedding(path=hparams.embeddings_path,
                                                        word_index=word_index,
                                                        embedding_dimensions=hparams.embed_dim,
                                                        trainable_embeddings=hparams.trainable_words,
                                                        embedding_index=embedding_index)

    char_matrix = generate_matrix(index=char_index, embedding_dimensions=hparams.char_dim)

    # Save the generated data
    save_json(train_contexts_path, train_contexts, format_json=False)
    save_json(train_answers_path, train_question_answers, format_json=False)
    save_json(dev_contexts_path, dev_contexts, format_json=False)
    save_json(dev_answers_path, dev_question_answers, format_json=False)
    # Save the word index mapping of word:index for both the pre-trained and trainable embeddings.
    save_json(word_index_path, word_index)
    save_json(char_index_path, char_index)
    save_json(trainable_index_path, trainable_word_index)
    # Save the trainable embeddings matrix.
    np.save(trainable_embeddings_path, trainable_matrix)
    # Save the full embeddings matrix
    np.save(word_embeddings_path, embedding_matrix)
    np.save(char_embeddings_path, char_matrix)
