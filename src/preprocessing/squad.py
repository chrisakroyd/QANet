import random
import re
from ftfy import fix_text
from tqdm import tqdm
from src.util import raw_data_paths, processed_data_paths, embedding_paths
from src.util import Tokenizer, generate_matrix, load_embeddings, save_embeddings, save_json, load_json, index_from_list

# Regexes
apostrophe = re.compile(r"('')")
apostrophe_like = re.compile(r"(``)")


def clean(text):
    text = fix_text(text)
    text = apostrophe.sub('" ', text)
    text = apostrophe_like.sub('" ', text)
    return text


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


def fit_on_squad(data_set, tokenizer):
    for data in tqdm(data_set['data']):
        for question_answer in data['paragraphs']:
            tokenizer.fit_on_texts(clean(question_answer['context']))
            for qa in question_answer['qas']:
                tokenizer.fit_on_texts(clean(qa['question']))
    return tokenizer


def pre_process(squad, tokenizer, hparams):
    elmo, indexed = [], []
    contexts, answers = {}, {}
    context_id, answer_id = 1, 1

    for data in tqdm(squad['data']):
        for question_answer in data['paragraphs']:
            context = clean(question_answer['context'])
            context_words, context_chars, _ = tokenizer.texts_to_sequences(context,
                                                                           max_words=hparams.context_limit,
                                                                           max_chars=hparams.char_limit,
                                                                           numpy=False, pad=False)
            # Tokenizer wraps in a list (len==1) so we need the last entry
            context_words = context_words[-1]
            context_chars = context_chars[-1]

            # Skip if outside of context limit
            if len(context_words) > hparams.context_limit:
                continue

            spans = convert_idx(context, tokenizer.tokenize(context))

            for qa in question_answer['qas']:
                question = clean(qa['question'])
                question_words, question_chars, _ = tokenizer.texts_to_sequences(question,
                                                                                 max_words=hparams.question_limit,
                                                                                 max_chars=hparams.char_limit,
                                                                                 numpy=False, pad=False)
                # Tokenizer wraps in a list (len==1) so we need the last entry
                question_words = question_words[-1]
                question_chars = question_chars[-1]

                # Skip if its outside of the limits.
                if len(question_words) > hparams.question_limit:
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

                if (answer_ends[-1] - answer_starts[-1]) > hparams.ans_limit:
                    continue

                indexed.append({
                    'context_words': context_words,
                    'context_chars': context_chars,
                    'question_words': question_words,
                    'question_chars': question_chars,
                    'answer_starts': answer_starts[-1],
                    'answer_ends': answer_ends[-1],
                    'answer_id': answer_id,
                })
                # For elmo we just need the tokens as opposed to word lookup indexes
                elmo.append({
                    'context_id': context_id,
                    'context_words': tokenizer.tokenize(context),
                    'question_words': tokenizer.tokenize(question),
                    'answer_starts': answer_starts[-1],
                    'answer_ends': answer_ends[-1],
                    'answer_id': answer_id,
                })

                answers[answer_id] = {
                    'id': qa['id'],
                    'answer_id': answer_id,
                    'context_id': context_id,
                    'answers': answer_texts,
                }

                answer_id += 1

            contexts[context_id] = {
                'id': context_id,
                'context': context,
                'word_spans': spans,
            }

            context_id += 1

    random.shuffle(indexed)
    print("{} questions in total".format(len(indexed)))
    return indexed, elmo, contexts, answers


def process(paths, hparams, vocab_path=None):
    train_path, dev_path = raw_data_paths(hparams)
    train_indexed_path, train_elmo_path, train_contexts_path, train_answers_path, \
    dev_indexed_path, dev_elmo_path, dev_contexts_path, dev_answers_path = processed_data_paths(hparams)
    # Get paths for saving embedding related info.
    word_index_path, char_index_path, trainable_index_path, trainable_embeddings_path,\
    word_embeddings_path, char_embeddings_path = embedding_paths(hparams)

    vocab = load_json(vocab_path) if vocab_path is not None else None

    tokenizer = Tokenizer(max_words=hparams.max_words + 1,
                          max_chars=hparams.max_chars + 1,
                          vocab=vocab,
                          lower=False,
                          oov_token=hparams.oov_token,
                          min_word_occurrence=hparams.min_word_ocur,
                          min_char_occurrence=hparams.min_char_ocur,
                          trainable_words=hparams.trainable_words,
                          filters=' ')  # Only filter stray whitespace

    train = load_json(train_path)
    dev = load_json(dev_path)

    print('Fitting...')
    tokenizer = fit_on_squad(train, tokenizer)
    tokenizer = fit_on_squad(dev, tokenizer)

    print('Processing...')
    train_indexed, train_elmo, train_contexts, train_answers = pre_process(train, tokenizer, hparams)
    dev_indexed, dev_elmo, dev_contexts, dev_answers = pre_process(dev, tokenizer, hparams)

    word_index = tokenizer.word_index
    char_index = tokenizer.char_index
    trainable_word_index = index_from_list(hparams.trainable_words)

    embedding_matrix, trainable_matrix = load_embeddings(path=hparams.embeddings_path,
                                                         word_index=word_index,
                                                         embedding_dimensions=hparams.embed_dim,
                                                         trainable_embeddings=hparams.trainable_words)

    char_matrix = generate_matrix(index=char_index, embedding_dimensions=hparams.char_dim)

    # Save the generated data
    save_json(train_indexed_path, train_indexed, format_json=False)
    save_json(train_elmo_path, train_elmo, format_json=False)
    save_json(train_contexts_path, train_contexts, format_json=False)
    save_json(train_answers_path, train_answers)
    save_json(dev_indexed_path, dev_indexed, format_json=False)
    save_json(dev_elmo_path, dev_elmo, format_json=False)
    save_json(dev_contexts_path, dev_contexts, format_json=False)
    save_json(dev_answers_path, dev_answers)
    # Save the word index mapping of word:index for both the pre-trained and trainable embeddings.
    save_json(word_index_path, word_index)
    save_json(char_index_path, char_index)
    save_json(trainable_index_path, trainable_word_index)
    # Save the trainable embeddings matrix.
    save_embeddings(trainable_embeddings_path, trainable_matrix, trainable_word_index)
    # Save the full embeddings matrix
    save_embeddings(word_embeddings_path, embedding_matrix, word_index)
    save_embeddings(char_embeddings_path, char_matrix, char_index)
