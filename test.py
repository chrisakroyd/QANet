import tensorflow as tf
from tqdm import tqdm
from src import config, constants, loaders, metrics, models, pipeline, train_utils, util


def test(sess_config, params):
    out_dir, model_dir, log_dir = util.train_paths(params)
    word_index_path, _, char_index_path = util.index_paths(params)
    embedding_paths = util.embedding_paths(params)

    _, _, _, _, test_context_path, test_answer_path = util.processed_data_paths(params)
    test_spans, test_answer_texts, test_ctxt_mapping = loaders.load_squad_v1_set(test_context_path, test_answer_path)
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
        # Create and initialize the model
        qanet = models.QANet(word_matrix, character_matrix, trainable_matrix, params)
        placeholders = iterator.get_next()
        is_training = tf.placeholder_with_default(True, shape=())
        qanet_inputs = util.dict_keys_as_tuple(placeholders, keys=constants.PlaceholderKeys.INPUT_KEYS)
        id_tensor = util.dict_keys_as_tuple(placeholders, keys=constants.PlaceholderKeys.ID_KEY)[0]
        start_logits, end_logits, start_pred, end_pred, _, _ = qanet(qanet_inputs, training=is_training)

        sess.run(tf.global_variables_initializer())
        # Restore the moving average version of the learned variables for eval.
        saver = train_utils.get_saver(ema_decay=params.ema_decay, ema_vars_only=True)
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        preds = []
        # +1 for uneven batch values, +1 for the range.
        for _ in tqdm(range(1, (len(test_answer_texts) // params.batch_size + 1) + 1)):
            answer_ids, answer_starts, answer_ends = sess.run([id_tensor, start_pred, end_pred],
                                                              feed_dict={is_training: False})
            preds.append((answer_ids, 0.0, answer_starts, answer_ends,))
        # Evaluate the predictions and reset the train result list for next eval period.
        eval_metrics, answer_texts = metrics.evaluate_list(preds, test_spans, test_answer_texts, test_ctxt_mapping)
        print("Exact Match: {}, F1: {}".format(eval_metrics['exact_match'], eval_metrics['f1']))

        if params.write_answer_file:
            results_path = util.results_path(params)
            out_file = {}
            for key, value in answer_texts.items():
                out_file[test_answers[key]['id']] = answer_texts[key]['prediction']
            util.save_json(results_path, out_file)


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.DEFAULTS)
    test(config.gpu_config(), config.model_config(defaults).FLAGS)
