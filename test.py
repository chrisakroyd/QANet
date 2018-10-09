import tensorflow as tf
from tqdm import tqdm
from src.config import gpu_config, model_config
from src.constants import FilePaths
from src.loaders import load_squad
from src.metrics import evaluate_list
from src.pipeline import create_dataset
from src.qa_net import QANet
from src.util import namespace_json, load_embeddings, train_paths, embedding_paths


def test(config, hparams):
    data_directory, model_directory, log_directory = train_paths(hparams)
    word_index_path, word_embedding_path, trainable_index_path, trainable_embedding_path, char_index_path, \
    character_embedding_path = embedding_paths(hparams)

    _, val = load_squad(hparams)
    val_contexts, val_spans, val_questions, val_answers, val_ctxt_mapping = val

    word_matrix, trainable_matrix, character_matrix = load_embeddings(
        index_paths=(word_index_path, trainable_index_path, char_index_path,),
        embedding_paths=(word_embedding_path, trainable_embedding_path, character_embedding_path),
        embed_dim=hparams.embed_dim,
        char_dim=hparams.char_dim
    )

    with tf.device('/cpu:0'):
        val_set, val_iter = create_dataset(val_contexts, val_questions, val_ctxt_mapping, hparams, shuffle=False)

    with tf.Session(config=config) as sess:
        # Create the dataset iterators.
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, val_set.output_types, val_set.output_shapes)
        # Create and initialize the model
        model = QANet(word_matrix, character_matrix, trainable_matrix, hparams)
        model.init(iterator.get_next())
        sess.run(tf.global_variables_initializer())
        val_handle = sess.run(val_iter.string_handle())
        # Restore the moving average version of the learned variables for eval.
        if hparams.ema_decay > 0.0:
            variable_averages = tf.train.ExponentialMovingAverage(0.)
            saver = tf.train.Saver(variable_averages.variables_to_restore())
        else:
            saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_directory))
        # Assign the shadow EMA variables to the graph.
        preds = []
        # +1 for uneven batch values, +1 for the range.
        for _ in tqdm(range(1, (len(val_answers) // hparams.batch_size + 1) + 1)):
            answer_ids, loss, answer_starts, answer_ends = sess.run(
                [model.answer_id, model.loss, model.yp1, model.yp2], feed_dict={handle: val_handle})
            preds.append((answer_ids, loss, answer_starts, answer_ends,))
        # Evaluate the predictions and reset the train result list for next eval period.
        metrics, answer_texts = evaluate_list(preds, val_spans, val_answers, val_ctxt_mapping)
        print("Exact Match: {}, F1: {}".format(metrics['exact_match'], metrics['f1']))

        if hparams.write_answer_file:
            out_file = {}
            for key, value in answer_texts.items():
                answer_id = key
                out_file[val_answers[answer_id]['id']] = answer_texts[key]['prediction']


if __name__ == '__main__':
    defaults = namespace_json(path=FilePaths.defaults.value)
    test(gpu_config(), model_config(defaults))
