import tensorflow as tf
from tqdm import tqdm
from src.config import gpu_config, model_config
from src.constants import FilePaths
from src.loaders import load_squad
from src.metrics import evaluate_list
from src.models import create_placeholders, create_dataset
from src.QANet import QANet
from src.util import namespace_json, load_contextual_embeddings, train_paths, embedding_paths


def test(config, hparams):
    data_directory, model_directory, log_directory = train_paths(hparams)
    word_index_path, word_embedding_path, char_index_path, character_embedding_path, trainable_index_path, \
    trainable_embedding_path, = embedding_paths(hparams)

    _, val = load_squad(hparams)
    val, val_contexts, val_answers = val

    word_matrix, trainable_matrix, character_matrix = load_contextual_embeddings(
        index_paths=(word_index_path, trainable_index_path, char_index_path,),
        embedding_paths=(word_embedding_path, trainable_embedding_path, character_embedding_path),
        embed_dim=hparams.embed_dim,
        char_dim=hparams.char_dim
    )

    placeholders = create_placeholders(hparams.context_limit, hparams.question_limit, hparams.char_limit)

    with tf.device('/cpu:0'):
        val_set, val_feed_dict = create_dataset(val, placeholders, batch_size=hparams.batch_size, shuffle=False)

    with tf.Session(config=config) as sess:
        # Create the dataset iterators.
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, val_set.output_types, val_set.output_shapes)
        val_iterator = val_set.make_initializable_iterator()
        # Create and initialize the model
        model = QANet(word_matrix, character_matrix, trainable_matrix, hparams)
        model.init(iterator.get_next())
        sess.run(tf.global_variables_initializer())
        sess.run(val_iterator.initializer, feed_dict=val_feed_dict)
        val_handle = sess.run(val_iterator.string_handle())
        # Restore the model
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_directory))
        # Assign the shadow EMA variables to the graph.
        sess.run(model.assign_vars)
        results = []
        # +1 for uneven batch values, +1 for the range.
        for _ in tqdm(range(1, (len(val_answers) // hparams.batch_size + 1) + 1)):
            answer_ids, loss, answer_starts, answer_ends = sess.run(
                [model.answer_id, model.loss, model.yp1, model.yp2], feed_dict={handle: val_handle})
            results.append((answer_ids, loss, answer_starts, answer_ends,))
        # Evaluate the predictions and reset the train result list for next eval period.
        metrics, answer_texts = evaluate_list(results, val_contexts, val_answers)

        print("Exact Match: {}, F1: {}".format(metrics['exact_match'], metrics['f1']))

        if hparams.write_answer_file:
            out_file = {}
            for key, value in answer_texts.items():
                answer_id = key
                out_file[val_answers[answer_id]['id']] = answer_texts[key]['prediction']


if __name__ == '__main__':
    defaults = namespace_json(path=FilePaths.defaults.value)
    test(gpu_config(), model_config(defaults))
