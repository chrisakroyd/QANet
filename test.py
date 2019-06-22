import operator
import tensorflow as tf
from tqdm import tqdm
from src import config, constants, loaders, metrics, models, pipeline, train_utils, util


def test(sess_config, params, checkpoint_selection=False):
    """
        Test procedure, optionally allows automated eval + ranking of all model checkpoints to find the best performing.
        Args:
            sess_config: tf session.
            params: hparams
            checkpoint_selection: Whether or not to run automated eval over all checkpoints as opposed to latest only.
    """
    model_dir, log_dir = util.save_paths(params)
    word_index_path, _, char_index_path = util.index_paths(params)
    embedding_paths = util.embedding_paths(params)

    _, _, _, _, test_context_path, test_answer_path = util.processed_data_paths(params)
    test_spans, test_answer_texts, test_ctxt_mapping = loaders.load_squad_set(test_context_path, test_answer_path)
    test_answers = util.load_json(test_answer_path)

    vocabs = util.load_vocab_files(paths=(word_index_path, char_index_path))
    word_matrix, trainable_matrix, character_matrix = util.load_numpy_files(paths=embedding_paths)

    with tf.device('/cpu:0'):
        tables = pipeline.create_lookup_tables(vocabs)
        _, _, test_record_path = util.tf_record_paths(params)
        test_data, iterator = pipeline.create_pipeline(params, tables, test_record_path, training=False)

    with tf.Session(config=sess_config) as sess:
        sess.run(iterator.initializer)
        sess.run(tf.tables_initializer())

        if params.use_contextual:
            qanet = models.QANetContextual(word_matrix, character_matrix, trainable_matrix, params)
        else:
            qanet = models.QANet(word_matrix, character_matrix, trainable_matrix, params)

        placeholders = iterator.get_next()
        is_training = tf.placeholder_with_default(True, shape=())
        start_logits, end_logits, start_pred, end_pred, _, _ = qanet(placeholders, training=is_training)
        id_tensor = util.unpack_dict(placeholders, keys=constants.PlaceholderKeys.ID_KEY)[0]

        sess.run(tf.global_variables_initializer())
        # Restore the moving average version of the learned variables for eval.
        saver = train_utils.get_saver(ema_decay=params.ema_decay, ema_vars_only=True)

        if checkpoint_selection:
            checkpoints = tqdm(tf.train.get_checkpoint_state(model_dir).all_model_checkpoint_paths)
        else:
            checkpoints = [tf.train.latest_checkpoint(model_dir)]

        results = []

        for checkpoint in checkpoints:
            saver.restore(sess, checkpoint)
            preds = []
            # +1 for uneven batch values, +1 for the range.
            for _ in tqdm(range(1, (len(test_answer_texts) // params.batch_size + 1) + 1)):
                answer_ids, answer_starts, answer_ends = sess.run([id_tensor, start_pred, end_pred],
                                                                  feed_dict={is_training: False})
                preds.append((answer_ids, 0.0, answer_starts, answer_ends,))
            # Evaluate the predictions and reset the train result list for next eval period.
            eval_metrics, answer_texts = metrics.evaluate_list(preds, test_spans, test_answer_texts, test_ctxt_mapping)
            em, f1 = util.unpack_dict(eval_metrics, keys=['exact_match', 'f1'])
            results.append({'exact_match': em, 'f1': f1, 'answer_texts': answer_texts, 'name': checkpoint})

        # In checkpoint selection mode we perform a search for
        if checkpoint_selection:
            # TODO: Cleanup this if statement.
            results = sorted(results, key=operator.itemgetter('exact_match'), reverse=True)
            print('\nCheckpoints ranked by Exact Match (EM):')
            for i, result in enumerate(results, start=1):
                print('{rank}: {name}, EM={em}'.format(rank=i,
                                                       name=util.filename(result['name']),
                                                       em=result['exact_match']))

            results = sorted(results, key=operator.itemgetter('f1'), reverse=True)
            print('\nCheckpoints ranked by F1 score:')
            for i, result in enumerate(results, start=1):
                print('{rank}: {name}, F1={f1}'.format(rank=i,
                                                       name=util.filename(result['name']),
                                                       f1=result['f1']))
        else:
            em, f1, answer_texts = util.unpack_dict(results[0], ['exact_match', 'f1', 'answer_texts'])
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
    test(config.gpu_config(), params)
