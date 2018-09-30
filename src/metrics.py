import numpy as np
import re
import string
import tensorflow as tf
from collections import Counter


def evaluate_list(results, contexts, answers, data_type, writer, step):
    answer_texts = {}
    losses = []

    for result in results:
        answer_ids, loss, answer_starts, answer_ends = result
        answer_data = get_answer_data(contexts, answers, answer_ids.tolist(),
                                      answer_starts.tolist(), answer_ends.tolist())
        answer_texts.update(answer_data)
        losses.append(loss)

    metrics = calc_metrics(answer_texts, losses, data_type, writer, step)

    return metrics


def calc_metrics(answer_texts, losses, data_type, writer, step):
    metrics = evaluate(answer_texts)
    metrics["loss"] = np.mean(losses)

    if writer is not None:
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="{}/loss".format(data_type),
                                                              simple_value=metrics["loss"]), ]), step)
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(
            tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ]), step)
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(
            tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ]), step)

    return metrics


def get_ground_truth_for_batch(contexts, answers, answer_ids):
    ground_truths = {}
    for answer_id in answer_ids:
        answer_id = str(answer_id)
        context_id = str(answers[answer_id]['context_id'])

        ground_truths[answer_id] = {
            'answer_id': answer_id,
            'context': contexts[context_id]['context'],
            'word_spans': contexts[context_id]['word_spans'],
            'answers': answers[answer_id]['answers']
        }
    return ground_truths


def get_answer_texts_for_batch(ground_truths, answer_ids, starts, ends):
    answer_texts = {}

    for answer_id, start, end in zip(answer_ids, starts, ends):
        answer_id = str(answer_id)
        context = ground_truths[answer_id]['context']
        spans = ground_truths[answer_id]['word_spans']
        # Get the text span from start to end and the corresponding ground truth.
        answer_texts[str(answer_id)] = {
            'prediction': context[spans[start][0]: spans[end][-1]],
            'ground_truth': ground_truths[answer_id]['answers']
        }

    return answer_texts


def get_answer_data(contexts, answers, answer_ids, answer_starts, answer_ends):
    ground_truths = get_ground_truth_for_batch(contexts, answers, answer_ids)
    answer_texts = get_answer_texts_for_batch(ground_truths, answer_ids, answer_starts, answer_ends)
    return answer_texts


def evaluate(answers):
    f1 = exact_match = total = 0
    for key, value in answers.items():
        total += 1

        ground_truths = answers[key]['ground_truth']
        prediction = answers[key]['prediction']
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score,
                                            prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
