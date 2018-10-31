import numpy as np
from src import util


def load_contexts(contexts):
    context_cache = {}
    context_spans = {}

    for key, value in contexts.items():
        context_words = value['context_tokens']
        context_cache[key] = context_words, len(value['context_words'])
        context_spans[key] = {
            'context': value['context'],
            'word_spans': value['word_spans'],
        }
    return context_cache, context_spans


def load_answers(answers):
    answer_cache = {}
    context_mapping = {}
    query_cache = {}

    for key, value in answers.items():
        query_words = value['query_tokens']
        answer_starts = np.asarray(value['answer_starts'], dtype=np.int32)
        answer_ends = np.asarray(value['answer_ends'], dtype=np.int32)

        answer_cache[key] = value['answers']
        context_mapping[key] = value['context_id']
        query_cache[key] = query_words, len(value['query_words']), answer_starts, answer_ends

    assert len(answer_cache) == len(answers)
    assert len(answer_cache) == len(query_cache)
    assert len(answer_cache) == len(context_mapping)

    return query_cache, answer_cache, context_mapping


def load_squad_set(contexts, answers):
    context_cache, contexts_spans = load_contexts(contexts)
    queries, answers, context_mapping = load_answers(answers)
    return context_cache, contexts_spans, queries, answers, context_mapping


def load_squad(hparams):
    train_context_path, train_answer_path, val_context_path, val_answer_path = util.processed_data_paths(hparams)
    train_context = util.load_json(train_context_path)
    train_answers = util.load_json(train_answer_path)
    val_context = util.load_json(val_context_path)
    val_answers = util.load_json(val_answer_path)

    train_set = load_squad_set(train_context, train_answers)
    val_set = load_squad_set(val_context, val_answers)

    return train_set, val_set
