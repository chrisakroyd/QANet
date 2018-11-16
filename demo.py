import os
import numpy as np
import tensorflow as tf
from src import config, constants, models, pipeline, train_utils, util


def demo(sess_config, params):
    # Although bad practice, I don't want to force people to install unnecessary dependencies to run this repo.
    from flask import Flask, json, request, send_from_directory
    app = Flask(__name__, static_folder=params.dist_dir)

    _, _, model_dir, _ = util.train_paths(params)
    word_index_path, _, char_index_path = util.index_paths(params)
    embedding_paths = util.embedding_paths(params)
    word_index = util.load_json(word_index_path)
    char_index = util.load_json(char_index_path)

    tokenizer = util.Tokenizer(lower=False,
                               oov_token=params.oov_token,
                               char_limit=params.char_limit,
                               word_index=word_index,
                               char_index=char_index,
                               trainable_words=params.trainable_words)

    vocabs = util.load_vocab_files(paths=(word_index_path, char_index_path))
    word_matrix, trainable_matrix, character_matrix = util.load_numpy_files(paths=embedding_paths)
    tables = pipeline.create_lookup_tables(vocabs)
    # Keep sess alive as long as the server is live, probably not best practice but it works @TODO Improve this.
    sess = tf.Session(config=sess_config)
    sess.run(tf.tables_initializer())
    qanet = models.QANet(word_matrix, character_matrix, trainable_matrix, params)
    pipeline_placeholders = pipeline.create_placeholders()
    demo_dataset, demo_iter = pipeline.create_demo_pipeline(params, tables, pipeline_placeholders)

    demo_placeholders = demo_iter.get_next()
    demo_inputs = train_utils.inputs_as_tuple(demo_placeholders)
    _, _, start_pred, end_pred, start_prob, end_prob = qanet(demo_inputs)
    demo_outputs = [start_pred, end_pred, start_prob, end_prob]
    sess.run(tf.global_variables_initializer())

    saver = train_utils.get_saver(ema_decay=params.ema_decay, ema_vars_only=True)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    @app.route('/predict/text', methods=['POST'])
    def process():
        data = request.get_json()
        context = data['context']
        query = data['query']

        context_tokens = tokenizer.tokenize(context)
        query_tokens = tokenizer.tokenize(query)
        # These values must match the names given to the input tensors in pipeline.py.
        # @TODO Fix this, there must be a better way of feeding values that is less fragile.
        sess.run(demo_iter.initializer, feed_dict={
            'context_tokens:0': np.array([context_tokens], dtype=np.str),
            'context_length:0': np.array([len(context_tokens)], dtype=np.int32),
            'query_tokens:0': np.array([query_tokens], dtype=np.str),
            'query_length:0': np.array([len(query_tokens)], dtype=np.int32),
        })

        try:
            answer_start, answer_end, p_start, p_end = sess.run(fetches=demo_outputs)
        except tf.errors.OutOfRangeError:
            # This in theory should never happen as we reset the iterator after each iteration and only run
            # one batch but theories are frequently wrong.
            raise RuntimeError('Iterator out of range, attempted to call too many times. (Please report this error)')

        response = {
            'contextTokens': context_tokens,
            'queryTokens': query_tokens,
            'answerTexts': 'NULL',  # @TODO Add in answer text extraction.
            'answerStarts': answer_start.tolist(),
            'answerEnds': answer_end.tolist(),
            'startProb': p_start.tolist(),
            'endProb': p_end.tolist(),
        }
        return json.dumps([response])

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != '' and os.path.exists(params.dist_dir + path):
            return send_from_directory(params.dist_dir, path)
        else:
            return send_from_directory(params.dist_dir, 'index.html')

    return app


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.defaults.value)
    app = demo(config.gpu_config(), config.model_config(defaults))
    app.run()
