import tensorflow as tf
import os
from tqdm import tqdm
from src import config, constants, loaders, metrics, models, pipeline, train_utils, util


def train(sess_config, params):
    # Get the directories where we save models+logs, create them if they do not exist for this run.
    _, out_dir, model_dir, log_dir = util.train_paths(params)
    word_index_path, _, char_index_path = util.index_paths(params)
    embedding_paths = util.embedding_paths(params)
    util.make_dirs([out_dir, model_dir, log_dir])

    train, val = loaders.load_squad(params)

    train_spans, train_answers, train_ctxt_mapping = train
    val_spans, val_answers, val_ctxt_mapping = val
    
    vocabs = util.load_vocab_files(paths=(word_index_path, char_index_path))
    word_matrix, trainable_matrix, character_matrix = util.load_numpy_files(paths=embedding_paths)

    with tf.device('/cpu:0'):
        tables = pipeline.create_lookup_tables(vocabs)

        train_tfrecords = util.tf_record_paths(params, training=True)
        val_tfrecords = util.tf_record_paths(params, training=False)
        train_set, train_iter = pipeline.create_pipeline(params, tables, train_tfrecords, training=True)
        _, val_iter = pipeline.create_pipeline(params, tables, val_tfrecords, training=False)

    with tf.Session(config=sess_config) as sess:
        sess.run([tf.tables_initializer(), train_iter.initializer, val_iter.initializer])
        # Create the dataset iterators.
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_set.output_types, train_set.output_shapes)
        # This section creates models, gets the output tensors and constructs the train_op. Although messy
        # it is written this way to bring it closer in line to the tf estimator API for easier later development.
        qanet = models.QANet(word_matrix, character_matrix, trainable_matrix, params)

        placeholders = iterator.get_next()
        # Features and labels.
        qanet_inputs = train_utils.inputs_as_tuple(placeholders)
        y_start, y_end, id_tensor = train_utils.labels_as_tuple(placeholders)

        start_logits, end_logits, start_pred, end_pred, _, _ = qanet(qanet_inputs, training=True)
        loss_op = qanet.compute_loss(start_logits, end_logits, y_start, y_end, l2=params.l2)

        train_op = train_utils.construct_train_op(loss_op,
                                                  learn_rate=params.learn_rate,
                                                  warmup_scheme=params.warmup_scheme,
                                                  warmup_steps=params.warmup_steps,
                                                  clip_norm=params.gradient_clip,
                                                  ema_decay=params.ema_decay,
                                                  beta1=params.beta1,
                                                  beta2=params.beta2,
                                                  epsilon=params.epsilon)
        # What ops we want the results of.
        train_outputs = [id_tensor, loss_op, start_pred, end_pred, train_op]
        val_outputs = [id_tensor, loss_op, start_pred, end_pred]
        sess.run(tf.global_variables_initializer())
        # Saver boilerplate
        writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
        saver = train_utils.get_saver()
        # Initialize the handles for switching.
        train_handle = sess.run(train_iter.string_handle())
        val_handle = sess.run(val_iter.string_handle())

        if os.path.exists(model_dir) and tf.train.latest_checkpoint(model_dir) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        global_step = max(sess.run(qanet.global_step), 1)
        train_preds = []

        for _ in tqdm(range(global_step, params.train_steps + 1)):
            global_step = sess.run(qanet.global_step) + 1
            # Either train + predict + save run metadata or train + predict
            if params.runtime_data and global_step % (params.checkpoint_every + 1) == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                answer_id, loss, answer_start, answer_end, _ = sess.run(fetches=train_outputs,
                                                                        feed_dict={handle: train_handle},
                                                                        options=run_options,
                                                                        run_metadata=run_metadata)

                writer.add_run_metadata(run_metadata, 'step%04d' % global_step)
                writer.flush()
            else:
                answer_id, loss, answer_start, answer_end, _ = sess.run(fetches=train_outputs,
                                                                        feed_dict={handle: train_handle})

            # Cache the result of the run for train evaluation.
            train_preds.append((answer_id, loss, answer_start, answer_end,))

            # Run the eval procedure, we use the predictions over train + eval and calculate EM + F1.
            if global_step % params.run_val_every == 0:
                val_preds = []
                # +1 for uneven batch values, +1 for the range.
                for _ in tqdm(range(1, (len(val_answers) // params.batch_size + 1) + 1)):
                    answer_id, loss, answer_start, answer_end = sess.run(fetches=val_outputs,
                                                                         feed_dict={handle: val_handle,
                                                                                    qanet.dropout: 0.0,
                                                                                    qanet.attn_dropout: 0.0,
                                                                                    })
                    val_preds.append((answer_id, loss, answer_start, answer_end,))
                # Evaluate the predictions and reset the train result list for next eval period.
                metrics.evaluate_list(train_preds, train_spans, train_answers, train_ctxt_mapping, 'train', writer,
                                      global_step)
                metrics.evaluate_list(val_preds, val_spans, val_answers, val_ctxt_mapping, 'val', writer,
                                      global_step)
                train_preds = []

            # Save the model weights.
            if global_step % params.checkpoint_every == 0:
                writer.flush()
                filename = os.path.join(model_dir, 'model_{}.ckpt'.format(global_step))
                # Save the model
                saver.save(sess, filename)


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.DEFAULTS)
    train(config.gpu_config(), config.model_config(defaults).FLAGS)
