from src import util


def load_contexts(contexts):
    context_spans = {}

    for key, value in contexts.items():
        context_spans[key] = {
            'context': value['context'],
            'word_spans': value['word_spans'],
        }

    return context_spans


def load_answers(answers):
    answer_cache = {}
    context_mapping = {}

    for key, value in answers.items():
        answer_cache[key] = value['answers']
        context_mapping[key] = value['context_id']

    assert len(answer_cache) == len(answers) == len(context_mapping)

    return answer_cache, context_mapping


def load_squad_set(contexts, answers):
    contexts_spans = load_contexts(contexts)
    answers, context_mapping = load_answers(answers)
    return contexts_spans, answers, context_mapping


def load_squad(hparams):
    train_context_path, train_answer_path, val_context_path, val_answer_path = util.processed_data_paths(hparams)
    train_context = util.load_json(train_context_path)
    train_answers = util.load_json(train_answer_path)
    val_context = util.load_json(val_context_path)
    val_answers = util.load_json(val_answer_path)

    train_set = load_squad_set(train_context, train_answers)
    val_set = load_squad_set(val_context, val_answers)

    return train_set, val_set
