import numpy as np
from src.util import load_json, pad_array, processed_data_paths


def pad(words, characters, word_limit, char_limit):
    padded_words = np.zeros((word_limit, ), dtype=np.int32)
    padded_chars = np.zeros((word_limit, char_limit, ), dtype=np.int32)

    padded_words[:len(words)] = words
    padded_chars[:len(words)] = [pad_array(char_arr, char_limit) for char_arr in characters]

    return padded_words, padded_chars


def load_squad_set(contexts, answers, hparams):
    data_size = len(answers)
    # Size of each dimension
    context_limit = hparams.context_limit
    question_limit = hparams.question_limit
    char_limit = hparams.char_limit
    # Init the arrays to hold the data.
    context_words = np.zeros((data_size, context_limit), dtype=np.int32)
    context_chars = np.zeros((data_size, context_limit, char_limit), dtype=np.int32)
    question_words = np.zeros((data_size, question_limit), dtype=np.int32)
    question_chars = np.zeros((data_size, question_limit, char_limit), dtype=np.int32)
    answer_starts = np.zeros((data_size, ), dtype=np.int32)
    answer_ends = np.zeros((data_size, ), dtype=np.int32)
    answer_ids = np.zeros((data_size, ), dtype=np.int32)
    # Pad the context words + chars and cache
    context_cache = {key: pad(value['context_words'], value['context_chars'], context_limit, char_limit)
                     for key, value in contexts.items()}

    for i, (key, row) in enumerate(answers.items()):
        context_id = str(row['context_id'])
        row_question_words = row['question_words']
        row_question_chars = row['question_chars']

        cached = context_cache[context_id]
        context_words[i] = np.copy(cached[0])
        context_chars[i] = np.copy(cached[-1])
        question_words[i], question_chars[i] = pad(row_question_words, row_question_chars, question_limit, char_limit)

        answer_starts[i] = row['answer_starts']
        answer_ends[i] = row['answer_ends']
        answer_ids[i] = row['answer_id']

        # Check that we have no entries of just 0.
        assert np.count_nonzero(context_words[i]) > 0
        assert np.count_nonzero(context_chars[i]) > 0
        assert np.count_nonzero(question_words[i]) > 0
        assert np.count_nonzero(question_chars[i]) > 0

    return context_words, context_chars, question_words, question_chars, answer_starts, answer_ends, answer_ids


def load_squad(hparams):
    train_context_path, train_answer_path, val_context_path, val_answer_path = processed_data_paths(hparams)
    train_context = load_json(train_context_path)
    train_answers = load_json(train_answer_path)
    val_context = load_json(val_context_path)
    val_answers = load_json(val_answer_path)

    train_set = load_squad_set(train_context, train_answers, hparams)
    val_set = load_squad_set(val_context, val_answers, hparams)

    return (train_set, train_context, train_answers), (val_set, val_context, val_answers)
