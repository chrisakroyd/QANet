import numpy as np
import tensorflow as tf
from collections import Counter
from src import preprocessing


def evaluate_list(preds, contexts, answers, context_mapping, data_type=None, writer=None, global_step=0):
    """ Calculates F1, EM and loss over a list of predictions.
        Args:
            preds: list of tuples containing start + end pointers, answer ids and loss.
            contexts: Context texts + word spans
            answers: answer_id: Ground truth mapping.
            context_mapping: answer_id: context_id mapping.
            data_type: String for whether we are in train/val.
            writer: Summary Writer object.
            global_step: Current step.
    """
    answer_texts = {}
    losses = []

    for pred in preds:
        answer_ids, loss, answer_starts, answer_ends = pred
        answer_data = get_answer_data(contexts, answers, context_mapping, answer_ids.tolist(),
                                      answer_starts.tolist(), answer_ends.tolist())
        answer_texts.update(answer_data)
        losses.append(loss)
    metrics = evaluate(answer_texts)
    metrics['loss'] = np.mean(losses)
    add_metric_summaries(metrics, data_type, writer, global_step)

    return metrics, answer_texts


def add_metric_summaries(metrics, data_type=None, writer=None, global_step=0):
    """ Adds summaries for various metric functions.
        Args:
            metrics: dict of metric_name: value.
            data_type: String for whether we are in train/val.
            writer: Summary Writer object.
            global_step: Current step.
    """
    if writer is not None and data_type is not None:
        for key, value in metrics.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag='{}/{}'.format(data_type, key), simple_value=value)])
            writer.add_summary(summ, global_step)


def get_answer_data(contexts, answers, context_mapping, answer_ids, answer_starts, answer_ends):
    """ Constructs a mapping of answer_ids to their answers and predictions.
        Args:
            contexts: Context texts + word spans
            answers: answer_id: Ground truth mapping.
            context_mapping: answer_id: context_id mapping.
            answer_ids:
            answer_starts: int start pointers.
            answer_ends: Int end pointers.
    """
    answer_texts = {}

    for answer_id, start, end in zip(answer_ids, answer_starts, answer_ends):
        answer_id = str(answer_id)
        context_id = str(context_mapping[answer_id])
        context = contexts[context_id]['context']
        spans = contexts[context_id]['word_spans']

        answer_texts[answer_id] = {
            'prediction': context[spans[start][0]: spans[end][-1]],
            'ground_truth': answers[answer_id]
        }

    return answer_texts


def evaluate(answers):
    """ Calculates all eval metrics taking the best over all answers. """
    f1 = exact_match = total = 0
    for key, value in answers.items():
        total += 1
        # Normalize the answers
        ground_truths = [preprocessing.normalize_answer(answer) for answer in answers[key]['ground_truth']]
        prediction = preprocessing.normalize_answer(answers[key]['prediction'])

        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def f1_score(prediction, ground_truth):
    """ Calculates F1 score, borrowed from SQuAD eval script. """
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """ Exact match is a simple as it sounds. """
    return prediction == ground_truth


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """ Can have multiple answers - We therefore take the best. """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
