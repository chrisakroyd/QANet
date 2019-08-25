import tensorflow as tf
import os
from tqdm import tqdm
from src import config, constants, loaders, metrics, models, pipeline, train_utils, util


def train(sess_config, params, debug=False):
    # Get the directories where we save models+logs, create them if they do not exist for this run.
    model_dir, log_dir = util.save_paths(params.models_dir, params.run_name)
    word_index_path, _, char_index_path = util.index_paths(params.data_dir, params.dataset)
    embedding_paths = util.embedding_paths(params.data_dir, params.dataset)
    util.make_dirs([model_dir, log_dir])

    use_contextual = params.model == constants.ModelTypes.QANET_CONTEXTUAL

    # Continue prompt for when params.heads is > 1, this can cause OOM so make sure its intentional.
    if params.heads > 1:
        if not util.yes_no_prompt(constants.Prompts.POSSIBLE_OOM.format(num_heads=params.heads)):
            exit(0)

    # Continue prompt for when we have a large buffer size, pre-embedded records are large and this can fill up RAM.
    if use_contextual and params.fixed_contextual_embeddings and params.shuffle_buffer_size > 10000:
        if not util.yes_no_prompt(constants.Prompts.LARGE_CONTEXTUAL_SHUFFLE_BUFFER):
            exit(0)

    util.save_config(params,
                     path=util.config_path(params.models_dir, params.run_name),
                     overwrite=False)  # Saves the run parameters in a .json

    train_data, val_data = loaders.load_squad(params)
    train_spans, train_answers, train_ctxt_mapping = train_data
    val_spans, val_answers, val_ctxt_mapping = val_data
    
    vocabs = util.load_vocab_files(paths=(word_index_path, char_index_path))
    word_matrix, trainable_matrix, character_matrix = util.load_numpy_files(paths=embedding_paths)
    num_val_batches = (len(val_answers) // params.batch_size + 1) + 1

    with tf.device('/cpu:0'):
        tables = pipeline.create_lookup_tables(vocabs)
        train_record_path, val_record_path, _ = util.tf_record_paths(params.data_dir, params.dataset)
        train_set, train_iter = pipeline.create_pipeline(params, tables, train_record_path,
                                                         use_contextual=use_contextual, training=True)
        _, val_iter = pipeline.create_pipeline(params, tables, val_record_path,
                                               use_contextual=use_contextual, training=False)

    with tf.Session(config=sess_config) as sess:
        sess.run([tf.tables_initializer(), train_iter.initializer, val_iter.initializer])
        # Create the dataset iterators, handles are required for switching between train/val modes within one graph.
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_set.output_types, train_set.output_shapes)

        qanet = models.create_model(word_matrix, character_matrix, trainable_matrix, params)
        placeholders = iterator.get_next()
        start_logits, end_logits, start_pred, end_pred, _, _ = qanet(placeholders, training=True)
        y_start, y_end, id_tensor = util.unpack_dict(placeholders, keys=constants.PlaceholderKeys.LABEL_KEYS)
        loss_op = qanet.compute_loss(start_logits, end_logits, y_start, y_end, l2=params.l2)

        train_op = train_utils.construct_train_op(loss_op,
                                                  learn_rate=params.learn_rate,
                                                  warmup_scheme=params.warmup_scheme,
                                                  warmup_steps=params.warmup_steps,
                                                  use_cosine_decay=params.use_cosine_decay,
                                                  plateau_steps=params.plateau_steps,
                                                  total_steps=params.train_steps,
                                                  clip_norm=params.gradient_clip,
                                                  ema_decay=params.ema_decay,
                                                  beta1=params.beta1,
                                                  beta2=params.beta2,
                                                  optimizer=params.optimizer,
                                                  weight_decay=params.weight_decay,
                                                  epsilon=params.epsilon)

        train_outputs = [id_tensor, loss_op, start_pred, end_pred, train_op]
        val_outputs = [id_tensor, loss_op, start_pred, end_pred]
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
        saver = train_utils.get_saver()

        train_handle = sess.run(train_iter.string_handle())
        val_handle = sess.run(val_iter.string_handle())

        if os.path.exists(model_dir) and tf.train.latest_checkpoint(model_dir) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        global_step = max(sess.run(qanet.global_step), 1)
        train_preds = []

        for _ in tqdm(range(global_step, params.train_steps + 1)):
            global_step = sess.run(qanet.global_step) + 1
            # In debug mode we record runtime metadata, e.g. Memory usage, performance
            # Refer to https://www.tensorflow.org/guide/graph_viz#runtime_statistics for more information.
            if debug and global_step % (params.checkpoint_every + 1) == 0:
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

            # We cache the result of each batch so we can compute metrics on the train set.
            train_preds.append((answer_id, loss, answer_start, answer_end, ))

            if global_step % params.checkpoint_every == 0:
                val_preds = []
                # +1 for uneven batch values, +1 for the range.
                for _ in tqdm(range(1, num_val_batches)):
                    answer_id, loss, answer_start, answer_end = sess.run(fetches=val_outputs,
                                                                         feed_dict={handle: val_handle,
                                                                                    qanet.dropout: 0.0,
                                                                                    qanet.attn_dropout: 0.0,
                                                                                    })
                    val_preds.append((answer_id, loss, answer_start, answer_end,))

                metrics.evaluate_list(train_preds, train_spans, train_answers, train_ctxt_mapping, 'train', writer,
                                      global_step, subsample_ratio=0.1)
                metrics.evaluate_list(val_preds, val_spans, val_answers, val_ctxt_mapping, 'val', writer,
                                      global_step)
                train_preds = []

                writer.flush()
                filename = os.path.join(model_dir, 'model_{}.ckpt'.format(global_step))
                saver.save(sess, filename)
                sess.run(val_iter.initializer)


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.DEFAULTS)
    flags = config.model_config(defaults).FLAGS
    params = util.load_config(flags, util.config_path(flags.models_dir, flags.run_name))  # Loads a pre-existing config otherwise == params
    train(config.gpu_config(), params)
