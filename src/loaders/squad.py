from src import util


def load_answers(answers):
    """ Loads answers into an answer dict and an answer_id: context_id lookup dict.
        Args:
            answers: dict containing answer_ids: answer_dict mappings.
        Returns:
            A dict containing answer_id: answer_text mappings and a dict for mapping answer_id: context_id.
    """
    answer_cache = {}
    context_mapping = {}

    for key, value in answers.items():
        answer_cache[key] = value['answers']
        context_mapping[key] = value['context_id']

    assert len(answer_cache) == len(answers) == len(context_mapping)

    return answer_cache, context_mapping


def load_squad_set(context_path, answer_path):
    """ Function that loads a processed squad dataset.
        Args:
            context_path: string filepath to a .json file containing context details.
            answer_path: string filepath to a .json file containing answers.
        Returns:
            A dict of context_ids mapping to a dict of context words and word spans, answers for eval and answer_id
            to context id mappings.
    """
    contexts = util.load_json(context_path)
    answers = util.load_json(answer_path)
    answers, context_mapping = load_answers(answers)
    return contexts, answers, context_mapping


def load_squad(hparams):
    """ Loads the processed train and dev squad sets.
        Args:
            hparams: A dictionary of parameters.
        Returns:
            Tuple of contexts, answers and context_mapping for both train and val.
    """
    train_context_path, train_answer_path, val_context_path, val_answer_path = util.processed_data_paths(hparams)
    train_set = load_squad_set(train_context_path, train_answer_path)
    val_set = load_squad_set(val_context_path, val_answer_path)

    return train_set, val_set
