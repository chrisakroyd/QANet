import tensorflow as tf
import os
from tqdm import tqdm
from src import config, constants, loaders, metrics, pipeline, util
from src.qanet import QANet


def train(sess_config, hparams):
    # Get the directories where we save models+logs, create them if they do not exist for this run.
    _, out_dir, model_dir, log_dir = util.train_paths(hparams)
    word_index_path, _, char_index_path = util.index_paths(hparams)
    word_embedding_path, trainable_embedding_path, char_embedding_path = util.embedding_paths(hparams)
    util.make_dirs([out_dir, model_dir, log_dir])

    train, val = loaders.load_squad(hparams)

    train_contexts, train_spans, train_queries, train_answers, train_ctxt_mapping = train
    val_contexts, val_spans, val_queries, val_answers, val_ctxt_mapping = val

    # Free some memory.
    del train
    del val

    word_vocab = util.load_vocab(path=word_index_path)
    char_vocab = util.load_vocab(path=char_index_path)

    word_matrix, trainable_matrix, character_matrix = util.load_embeddings(
        embedding_paths=(word_embedding_path, trainable_embedding_path, char_embedding_path),
    )

    with tf.device('/cpu:0'):
        if hparams.use_tf_record:
            train_args = util.tf_record_paths(hparams, train=True)
            val_args = util.tf_record_paths(hparams, train=False)
        else:
            train_args = train_contexts, train_queries, train_ctxt_mapping
            val_args = val_contexts, val_queries, val_ctxt_mapping

        train_set, train_iter = pipeline.create_pipeline(hparams, word_vocab, char_vocab, train_args, train=True)
        _, val_iter = pipeline.create_pipeline(hparams, word_vocab, char_vocab, val_args, train=False)

    with tf.Session(config=sess_config) as sess:
        sess.run(train_iter.initializer)
        sess.run(val_iter.initializer)
        sess.run(tf.tables_initializer())
        # Create the dataset iterators.
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_set.output_types, train_set.output_shapes)
        # Create and initialize the model
        model = QANet(word_matrix, character_matrix, trainable_matrix, hparams)
        model.init(iterator.get_next())
        sess.run(tf.global_variables_initializer())
        # saver boilerplate
        writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
        saver = tf.train.Saver()
        # Initialize the handles for switching.
        train_handle = sess.run(train_iter.string_handle())
        val_handle = sess.run(val_iter.string_handle())

        if os.path.exists(model_dir) and tf.train.latest_checkpoint(model_dir) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        global_step = max(sess.run(model.global_step), 1)
        train_preds = []

        for _ in tqdm(range(global_step, hparams.train_steps + 1)):
            global_step = sess.run(model.global_step) + 1
            # Either train + predict + save run metadata or train + predict
            if hparams.runtime_data and global_step % (hparams.checkpoint_every + 1) == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                answer_ids, loss, answer_start, answer_end, _ = sess.run([model.answer_id, model.loss,
                                                                          model.start_pointer, model.end_pointer,
                                                                          model.train_op],
                                                                         feed_dict={handle: train_handle},
                                                                         options=run_options,
                                                                         run_metadata=run_metadata)

                writer.add_run_metadata(run_metadata, 'step%03d' % global_step)
                writer.flush()
            else:
                answer_ids, loss, answer_start, answer_end, _ = sess.run([model.answer_id, model.loss,
                                                                          model.start_pointer, model.end_pointer,
                                                                          model.train_op],
                                                                         feed_dict={handle: train_handle})

            # Cache the result of the run for train evaluation.
            train_preds.append((answer_ids, loss, answer_start, answer_end,))

            if global_step % hparams.save_loss_every == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss)])
                writer.add_summary(loss_sum, global_step)

            # Run the eval procedure, we use the predictions over train + eval and calculate EM + F1.
            if global_step % hparams.run_val_every == 0:
                val_preds = []
                # +1 for uneven batch values, +1 for the range.
                for _ in tqdm(range(1, (len(val_answers) // hparams.batch_size + 1) + 1)):
                    answer_ids, loss, answer_start, answer_end = sess.run([model.answer_id, model.loss,
                                                                           model.start_pointer, model.end_pointer],
                                                                          feed_dict={handle: val_handle})
                    val_preds.append((answer_ids, loss, answer_start, answer_end,))
                # Evaluate the predictions and reset the train result list for next eval period.
                metrics.evaluate_list(train_preds, train_spans, train_answers, train_ctxt_mapping, 'train', writer,
                                      global_step)
                metrics.evaluate_list(val_preds, val_spans, val_answers, val_ctxt_mapping, 'val', writer,
                                      global_step)
                train_preds = []

            # Save the model weights.
            if global_step % hparams.checkpoint_every == 0:
                writer.flush()
                filename = os.path.join(model_dir, 'model_{}.ckpt'.format(global_step))
                # Save the model
                saver.save(sess, filename)


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.defaults.value)
    train(config.gpu_config(), config.model_config(defaults).FLAGS)
