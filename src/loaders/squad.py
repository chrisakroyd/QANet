import numpy as np
from src import util


# tf data api doesn't work well with un-padded nested structures, therefore pad the nested dimension to char limit.
def pad_chars(chars, limit):
    return np.asarray([util.pad_array(char_arr, limit) for char_arr in chars], dtype=np.int32)


def load_contexts(contexts, char_limit):
    context_cache = {}
    context_spans = {}

    for key, value in contexts.items():
        context_words = np.asarray(value['context_words'], dtype=np.int32)
        context_chars = pad_chars(value['context_chars'], char_limit)
        context_cache[key] = context_words, context_chars, len(value['context_words'])
        context_spans[key] = {
            'context': value['context'],
            'word_spans': value['word_spans'],
        }
    return context_cache, context_spans


def load_answers(answers, char_limit):
    answer_cache = {}
    context_mapping = {}
    query_cache = {}

    for key, value in answers.items():
        query_words = np.asarray(value['query_words'], dtype=np.int32)
        query_chars = pad_chars(value['query_chars'], char_limit)
        answer_starts = np.asarray(value['answer_starts'], dtype=np.int32)
        answer_ends = np.asarray(value['answer_ends'], dtype=np.int32)

        answer_cache[key] = value['answers']
        context_mapping[key] = value['context_id']
        query_cache[key] = query_words, query_chars, len(value['query_words']), answer_starts, answer_ends

    assert len(answer_cache) == len(answers)
    assert len(answer_cache) == len(query_cache)
    assert len(answer_cache) == len(context_mapping)

    return query_cache, answer_cache, context_mapping


def load_squad_set(contexts, answers, hparams):
    context_cache, contexts_spans = load_contexts(contexts, hparams.char_limit)
    queries, answers, context_mapping = load_answers(answers, hparams.char_limit)
    return context_cache, contexts_spans, queries, answers, context_mapping


def load_squad(hparams):
    train_context_path, train_answer_path, val_context_path, val_answer_path = util.processed_data_paths(hparams)
    train_context = util.load_json(train_context_path)
    train_answers = util.load_json(train_answer_path)
    val_context = util.load_json(val_context_path)
    val_answers = util.load_json(val_answer_path)

    train_set = load_squad_set(train_context, train_answers, hparams)
    val_set = load_squad_set(val_context, val_answers, hparams)

    return train_set, val_set
