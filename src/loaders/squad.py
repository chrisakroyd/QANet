import numpy as np
from src.util import load_json, pad_array, processed_data_paths


# tf data api doesn't work well with un-padded nested structures, therefore pad the nested dimension to char limit.
def pad_chars(chars, limit):
    return np.asarray([pad_array(char_arr, limit) for char_arr in chars], dtype=np.int32)


def load_contexts(contexts, char_limit):
    context_cache = {}
    context_spans = {}

    for key, value in contexts.items():
        context_words = np.asarray(value['context_words'], dtype=np.int32)
        context_chars = pad_chars(value['context_chars'], char_limit)
        context_cache[key] = context_words, context_chars
        context_spans[key] = {
            'context': value['context'],
            'word_spans': value['word_spans'],
        }
    return context_cache, context_spans


def load_answers(answers, char_limit):
    answer_cache = {}
    context_mapping = {}
    question_cache = {}

    for key, value in answers.items():
        question_words = np.asarray(value['question_words'], dtype=np.int32)
        question_chars = pad_chars(value['question_chars'], char_limit)
        answer_starts = np.asarray(value['answer_starts'], dtype=np.int32)
        answer_ends = np.asarray(value['answer_ends'], dtype=np.int32)

        answer_cache[key] = value['answers']
        context_mapping[key] = value['context_id']
        question_cache[key] = question_words, question_chars, answer_starts, answer_ends

    return question_cache, answer_cache, context_mapping


def load_squad_set(contexts, answers, hparams):
    context_cache, contexts_spans = load_contexts(contexts, hparams.char_limit)
    questions, answers, context_mapping = load_answers(answers, hparams.char_limit)
    return context_cache, contexts_spans, questions, answers, context_mapping


def load_squad(hparams):
    train_context_path, train_answer_path, val_context_path, val_answer_path = processed_data_paths(hparams)
    train_context = load_json(train_context_path)
    train_answers = load_json(train_answer_path)
    val_context = load_json(val_context_path)
    val_answers = load_json(val_answer_path)

    train_set = load_squad_set(train_context, train_answers, hparams)
    val_set = load_squad_set(val_context, val_answers, hparams)

    return train_set, val_set
