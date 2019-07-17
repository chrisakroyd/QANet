import itertools
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from src import config, constants, loaders, metrics, models, pipeline, train_utils, util


def pointer_prob(probs, indices, axis=1):
    """
        Given a vector of indices (predicted pointers) of length b and a list of probabilities of shape b * N,
        returns a vector of length b with the corresponding probabilities for that pointer.

        This is a pre-step to scoring and ranking models based on their span probability and is used as part of the
        ensemble_best strategy.

        Args:
            probs: A b * N matrix where N is the length of the sequence and b is the batch size used.
            indices: A vector of length b consisting of integer pointers.
            axis: Integer, axis to reduce on.
        Returns:
            A floating point vector of length b.
    """
    indices = np.expand_dims(indices, axis=axis)
    return np.squeeze(np.take_along_axis(probs, indices, axis=axis))


def calculate_scores(predictions):
    """
        Calculates a score for each prediction, in the current version it is the product of the start and end
        probabilities.

        Args:
            predictions: List of tuples, with each tuple being in the form
                        (answer_ids, answer_starts, answer_ends, prob_starts, prob_ends)
        Returns:
            Four n-length arrays where n is the length of the dataset in the order: answer_ids, answer_starts,
            answer_ends, scores.
    """
    # Converts from N rows of length C to C columns of length N
    answer_ids, answer_starts, answer_ends, prob_starts, prob_ends = zip(*predictions)
    scores = [pointer_prob(p_start, ans_start) * pointer_prob(p_end, ans_end)
              for ans_start, ans_end, p_start, p_end in zip(answer_starts, answer_ends, prob_starts, prob_ends)]
    # Instead of having n batches we concat all batches together in order to save a few lines of code down the line.
    answer_ids, answer_starts, answer_ends, scores = map(np.concatenate,
                                                         [answer_ids, answer_starts, answer_ends, scores])
    return answer_ids, answer_starts, answer_ends, scores


def ensemble_best(model_predictions):
    """
        Ensemble method that chooses the most probable answer from a set of predictions. Able to handle an arbitrary
        number of models.
        (Models aren't mixed, answer comes from the single model with the highest probability)

        Args:
            model_predictions:
        Returns:
            Best predictions
    """
    answer_ids, answer_starts, answer_ends, scores = zip(*[calculate_scores(pred) for pred in model_predictions])
    # This check ensures all the ids in the answer_id arrays line up for each set of predictions, if this fails
    # check that iterators are reset after each test run.
    assert all(map(lambda x: np.array_equal(*x), itertools.combinations(answer_ids, r=2)))
    answer_ids = answer_ids[-1]  # After we have asserted they are all equal, just grab any of the id arrays.

    # As we operate over the same dataset in the same order each time we can stack all model predictions and argmax
    # on the first dimension to get an n (n = length of dataset) sized array of indices for the model that has
    # the best prediction for each answer. After this, we enumerate over the n sized array and pick out the
    # corresponding best answer.
    answer_starts, answer_ends, scores = map(np.stack, [answer_starts, answer_ends, scores])
    model_indices = np.argmax(scores, axis=0)

    assert len(model_indices) == len(answer_ids)

    answer_starts = [answer_starts[m, i] for i, m in enumerate(model_indices)]
    answer_ends = [answer_ends[m, i] for i, m in enumerate(model_indices)]

    assert len(answer_ids) == len(answer_starts) == len(answer_ends)

    return answer_ids, answer_starts, answer_ends


def ensemble(sess_config, params, checkpoint_ensemble=False):
    """
        Test procedure, optionally allows automated eval + ranking of all model checkpoints to find the best performing.
        Args:
            sess_config: tf session.
            params: hparams
            checkpoint_ensemble: Whether or not to run automated eval over all checkpoints as opposed to latest only.
    """

    word_index_path, _, char_index_path = util.index_paths(params)
    embedding_paths = util.embedding_paths(params)

    _, _, _, _, test_context_path, test_answer_path = util.processed_data_paths(params)
    test_spans, test_answer_texts, test_ctxt_mapping = loaders.load_squad_set(test_context_path, test_answer_path)
    test_answers = util.load_json(test_answer_path)
    vocabs = util.load_vocab_files(paths=(word_index_path, char_index_path))
    word_matrix, trainable_matrix, character_matrix = util.load_numpy_files(paths=embedding_paths)

    model_dir, log_dir = util.save_paths(params)
    use_contextual = params.model == constants.ModelTypes.QANET_CONTEXTUAL

    with tf.device('/cpu:0'):
        tables = pipeline.create_lookup_tables(vocabs)
        _, _, test_record_path = util.tf_record_paths(params)
        _, iterator = pipeline.create_pipeline(params, tables, test_record_path,
                                               use_contextual=use_contextual, training=False)

    with tf.Session(config=sess_config) as sess:
        sess.run(iterator.initializer)
        sess.run(tf.tables_initializer())

        if params.model == constants.ModelTypes.QANET:
            qanet = models.QANet(word_matrix, character_matrix, trainable_matrix, params)
        elif params.model == constants.ModelTypes.QANET_CONTEXTUAL:
            qanet = models.QANetContextual(word_matrix, character_matrix, trainable_matrix, params)
        else:
            raise ValueError('Unsupported model type.')

        placeholders = iterator.get_next()
        is_training = tf.placeholder_with_default(True, shape=())
        start_logits, end_logits, start_pred, end_pred, start_prob, end_prob = qanet(placeholders, training=is_training)
        id_tensor = util.unpack_dict(placeholders, keys=constants.PlaceholderKeys.ID_KEY)

        sess.run(tf.global_variables_initializer())
        # Restore the moving average version of the learned variables for eval.
        saver = train_utils.get_saver(ema_decay=params.ema_decay, ema_vars_only=True)

        if checkpoint_ensemble:
            checkpoints = tqdm(tf.train.get_checkpoint_state(model_dir).all_model_checkpoint_paths)
        else:
            raise ValueError('Not implemented.')

        model_predictions = []

        for checkpoint in checkpoints:
            saver.restore(sess, checkpoint)
            preds = []
            # +1 for uneven batch values, +1 for the range.
            for _ in tqdm(range(1, (len(test_answer_texts) // params.batch_size + 1) + 1)):
                answer_ids, answer_starts, answer_ends, prob_starts, prob_ends = sess.run(
                    [id_tensor, start_pred, end_pred, start_prob, end_prob],
                    feed_dict={is_training: False})

                preds.append((answer_ids, answer_starts, answer_ends, prob_starts, prob_ends,))
            model_predictions.append(preds)
            sess.run(iterator.initializer)  # Resets val iterator, guarantees that

        answer_ids, answer_starts, answer_ends = ensemble_best(model_predictions)
        answer_texts = metrics.get_answer_data(test_spans, test_answer_texts, test_ctxt_mapping,
                                               answer_ids, answer_starts, answer_ends)
        eval_metrics = metrics.evaluate(answer_texts)
        em, f1 = util.unpack_dict(eval_metrics, keys=['exact_match', 'f1'])

        print('\nExact Match: {em}, F1: {f1}'.format(em=em, f1=f1))

        if params.write_answer_file:
            results_path = util.results_path(params)
            out_file = {}
            for key, value in answer_texts.items():
                out_file[test_answers[key]['id']] = answer_texts[key]['prediction']
            util.save_json(results_path, out_file)


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.DEFAULTS)
    flags = config.model_config(defaults).FLAGS
    params = util.load_config(flags, util.config_path(flags))  # Loads a pre-existing config otherwise == params
    ensemble(config.gpu_config(), params)
