import tensorflow as tf
import os
from tqdm import tqdm
from src.config import gpu_config, model_config
from src.constants import FilePaths
from src.loaders import load_squad
from src.metrics import evaluate_list
from src.pipeline import create_dataset
from src.QANet import QANet
from src.util import namespace_json, load_embeddings, make_dirs, train_paths, embedding_paths


def train(config, hparams):
    # Get the directories where we save models+logs, create them if they do not exist for this run.
    data_directory, model_directory, log_directory = train_paths(hparams)
    word_index_path, word_embedding_path, trainable_index_path, trainable_embedding_path, char_index_path, \
    character_embedding_path = embedding_paths(hparams)
    make_dirs([model_directory, log_directory])

    train, val = load_squad(hparams)

    train_contexts, train_spans, train_questions, train_answers, train_ctxt_mapping = train
    val_contexts, val_spans, val_questions, val_answers, val_ctxt_mapping = val

    word_matrix, trainable_matrix, character_matrix = load_embeddings(
        index_paths=(word_index_path, trainable_index_path, char_index_path, ),
        embedding_paths=(word_embedding_path, trainable_embedding_path, character_embedding_path, ),
        embed_dim=hparams.embed_dim,
        char_dim=hparams.char_dim
    )

    with tf.device('/cpu:0'):
        train_set, train_iter = create_dataset(train_contexts, train_questions, train_ctxt_mapping, hparams)
        _, val_iter = create_dataset(val_contexts, val_questions, val_ctxt_mapping, hparams, shuffle=False)

    with tf.Session(config=config) as sess:
        # Create the dataset iterators.
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_set.output_types, train_set.output_shapes)
        # Create and initialize the model
        model = QANet(word_matrix, character_matrix, trainable_matrix, hparams)
        model.init(iterator.get_next())
        sess.run(tf.global_variables_initializer())
        # saver boilerplate
        writer = tf.summary.FileWriter(log_directory, graph=sess.graph)
        saver = tf.train.Saver()
        # Initialize the handles for switching.
        train_handle = sess.run(train_iter.string_handle())
        val_handle = sess.run(val_iter.string_handle())

        if os.path.exists(model_directory) and tf.train.latest_checkpoint(model_directory) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(model_directory))

        global_step = max(sess.run(model.global_step), 1)
        train_preds = []

        for _ in tqdm(range(global_step, hparams.train_steps)):
            global_step = sess.run(model.global_step) + 1
            # Either train + predict + save run metadata or train + predict
            if hparams.runtime_data and global_step % (hparams.checkpoint_every + 1) == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                answer_ids, loss, l2_loss, answer_start, answer_end, _ = sess.run(
                    [model.answer_id, model.loss, model.l2_loss, model.yp1, model.yp2, model.train_op],
                    feed_dict={handle: train_handle},
                    options=run_options,
                    run_metadata=run_metadata)

                writer.add_run_metadata(run_metadata, 'step%03d' % global_step)
                writer.flush()
            else:
                answer_ids, loss, l2_loss, answer_start, answer_end, _ = sess.run(
                    [model.answer_id, model.loss, model.l2_loss, model.yp1, model.yp2, model.train_op],
                    feed_dict={handle: train_handle})

            # Cache the result of the run for train evaluation.
            train_preds.append((answer_ids, loss, answer_start, answer_end,))

            # Save the loss + l2 loss
            if global_step % hparams.save_loss_every == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss)])
                reg_loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/l2_loss", simple_value=l2_loss)])
                writer.add_summary(loss_sum, global_step)
                writer.add_summary(reg_loss_sum, global_step)

            # Run the eval procedure, we use the predictions over train + eval and calculate EM + F1.
            if global_step % hparams.run_val_every == 0:
                val_preds = []
                # +1 for uneven batch values, +1 for the range.
                for _ in tqdm(range(1, (len(val_answers) // hparams.batch_size + 1) + 1)):
                    answer_ids, loss, answer_start, answer_end = sess.run(
                        [model.answer_id, model.loss, model.yp1, model.yp2], feed_dict={handle: val_handle})
                    val_preds.append((answer_ids, loss, answer_start, answer_end,))
                # Evaluate the predictions and reset the train result list for next eval period.
                evaluate_list(train_preds, train_spans, train_answers, train_ctxt_mapping, 'train', writer, global_step)
                evaluate_list(val_preds, val_spans, val_answers, val_ctxt_mapping, 'val', writer, global_step)
                train_preds = []

            # @TODO Add in saving trained embeddings.
            # Save the model weights.
            if global_step % hparams.checkpoint_every == 0:
                writer.flush()
                filename = os.path.join(model_directory, 'model_{}.ckpt'.format(global_step))
                # Save the model
                saver.save(sess, filename)


if __name__ == '__main__':
    defaults = namespace_json(path=FilePaths.defaults.value)
    train(gpu_config(), model_config(defaults).FLAGS)
