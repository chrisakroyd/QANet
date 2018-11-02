import tensorflow as tf
from tqdm import tqdm
from src import config, constants, loaders, metrics, pipeline, qanet, util


def test(sess_config, hparams):
    _, out_dir, model_dir, log_dir = util.train_paths(hparams)
    word_index_path, _, char_index_path = util.index_paths(hparams)
    word_index_path, word_embedding_path, trainable_index_path, trainable_embedding_path, char_index_path, \
    char_embedding_path = util.embedding_paths(hparams)

    _, val = loaders.load_squad(hparams)
    val_contexts, val_spans, val_queries, val_answers, val_ctxt_mapping = val

    word_vocab = util.load_vocab(path=word_index_path)
    char_vocab = util.load_vocab(path=char_index_path)
    word_matrix, trainable_matrix, character_matrix = util.load_embeddings(
        embedding_paths=(word_embedding_path, trainable_embedding_path, char_embedding_path),
    )

    with tf.device('/cpu:0'):
        word_table, char_table = pipeline.create_lookup_tables(word_vocab, char_vocab)
        if hparams.use_tf_record:
            val_args = util.tf_record_paths(hparams, train=False)
        else:
            val_args = val_contexts, val_queries, val_ctxt_mapping
        val_set, val_iter = pipeline.create_pipeline(hparams, word_table, char_table, val_args, train=False)

    with tf.Session(config=sess_config) as sess:
        # Create the dataset iterators.
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, val_set.output_types, val_set.output_shapes)
        # Create and initialize the model
        model = qanet.QANet(word_matrix, character_matrix, trainable_matrix, hparams)
        model.init(iterator.get_next())
        sess.run(tf.global_variables_initializer())
        val_handle = sess.run(val_iter.string_handle())
        # Restore the moving average version of the learned variables for eval.
        if hparams.ema_decay > 0.0:
            variable_averages = tf.train.ExponentialMovingAverage(0.)
            saver = tf.train.Saver(variable_averages.variables_to_restore())
        else:
            saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        # Assign the shadow EMA variables to the graph.
        preds = []
        # +1 for uneven batch values, +1 for the range.
        for _ in tqdm(range(1, (len(val_answers) // hparams.batch_size + 1) + 1)):
            answer_ids, answer_starts, answer_ends = sess.run([model.answer_id, model.start_pointer, model.end_pointer],
                                                              feed_dict={handle: val_handle})
            preds.append((answer_ids, 0.0, answer_starts, answer_ends,))
        # Evaluate the predictions and reset the train result list for next eval period.
        eval_metrics, answer_texts = metrics.evaluate_list(preds, val_spans, val_answers, val_ctxt_mapping)
        print("Exact Match: {}, F1: {}".format(eval_metrics['exact_match'], eval_metrics['f1']))

        if hparams.write_answer_file:
            out_file = {}
            for key, value in answer_texts.items():
                answer_id = key
                out_file[val_answers[answer_id]['id']] = answer_texts[key]['prediction']


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.defaults.value)
    test(config.gpu_config(), config.model_config(defaults))
