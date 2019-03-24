import os
import random
import numpy as np
import tensorflow as tf
from src import config, constants, demo_utils, models, pipeline, preprocessing as prepro, train_utils, util

API_VERSION = 1
BAD_REQUEST_CODE = 400


def demo(sess_config, params):
    # Although bad practice, I don't want to force people to install unnecessary dependencies to run this repo.
    from flask import Flask, json, request, send_from_directory

    # TODO This is a mess and shouldn't be here but is neccessary for demo_ui development.
    # Comes from https://gist.github.com/blixt/54d0a8bf9f64ce2ec6b8
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        if request.method == 'OPTIONS':
            response.headers['Access-Control-Allow-Methods'] = 'DELETE, GET, POST, PUT'
            headers = request.headers.get('Access-Control-Request-Headers')
            if headers:
                response.headers['Access-Control-Allow-Headers'] = headers
        return response

    app = Flask(__name__, static_folder=params.dist_dir)
    app.after_request(add_cors_headers)

    model_dir, _ = util.save_paths(params)
    word_index_path, _, char_index_path = util.index_paths(params)
    examples_path = util.examples_path(params)
    embedding_paths = util.embedding_paths(params)
    word_index, char_index, examples = util.load_multiple_jsons(paths=(word_index_path, char_index_path, examples_path))

    tokenizer = util.Tokenizer(lower=False,
                               oov_token=params.oov_token,
                               word_index=word_index,
                               char_index=char_index,
                               trainable_words=params.trainable_words,
                               filters=None)

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
    demo_inputs = util.dict_keys_as_tuple(demo_placeholders, keys=constants.PlaceholderKeys.INPUT_KEYS)
    _, _, start_pred, end_pred, start_prob, end_prob = qanet(demo_inputs)
    demo_outputs = [start_pred, end_pred, start_prob, end_prob]
    sess.run(tf.global_variables_initializer())

    saver = train_utils.get_saver(ema_decay=params.ema_decay, ema_vars_only=True)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    @app.route('/api/v{0}/model/predict'.format(API_VERSION), methods=['POST'])
    def predict():
        data = request.get_json()

        if 'context' not in data:
            return json.dumps(demo_utils.get_error_response(constants.ErrorMessages.NO_CONTEXT,
                                                            data, error_code=1)), BAD_REQUEST_CODE
        elif 'query' not in data:
            return json.dumps(demo_utils.get_error_response(constants.ErrorMessages.NO_QUERY,
                                                            data, error_code=2)), BAD_REQUEST_CODE

        context = prepro.normalize(data['context'])
        query = prepro.normalize(data['query'])

        context_tokens = tokenizer.tokenize(context)
        query_tokens = tokenizer.tokenize(query)

        if len(data['context']) <= 0 or len(context_tokens) <= 0:
            return json.dumps(demo_utils.get_error_response(constants.ErrorMessages.INVALID_CONTEXT,
                                                            data, error_code=3)), BAD_REQUEST_CODE
        elif len(data['query']) <= 0 or len(query_tokens) <= 0:
            return json.dumps(demo_utils.get_error_response(constants.ErrorMessages.INVALID_QUERY,
                                                            data, error_code=4)), BAD_REQUEST_CODE

        if len(context_tokens) > params.max_tokens:
            context_tokens, context_lengths = demo_utils.split_text(context_tokens, params.max_tokens)
            query_lengths = [len(query_tokens)] * len(context_lengths)
            query_tokens = [query_tokens] * len(context_lengths)
        else:
            context_lengths = [len(context_tokens)]
            query_lengths = [len(query_tokens)]
            context_tokens = [context_tokens]
            query_tokens = [query_tokens]

        answer_start, answer_end, p_start, p_end = process(context_tokens, context_lengths, query_tokens, query_lengths)
        response = demo_utils.get_predict_response(context_tokens, query_tokens, answer_start,
                                                   answer_end, p_start, p_end, data)

        return json.dumps(response)

    def process(context_tokens, context_lengths, query_tokens, query_lengths):
        # These values must match the names given to the input tensors in pipeline.py.
        # @TODO Fix this, there must be a better way of feeding values that is less fragile.
        sess.run(demo_iter.initializer, feed_dict={
            'context_tokens:0': np.array(context_tokens, dtype=np.str),
            'context_length:0': np.array(context_lengths, dtype=np.int32),
            'query_tokens:0': np.array(query_tokens, dtype=np.str),
            'query_length:0': np.array(query_lengths, dtype=np.int32),
        })

        answer_start, answer_end, p_start, p_end = sess.run(fetches=demo_outputs)
        return answer_start, answer_end, p_start, p_end

    @app.route('/api/v{0}/examples'.format(API_VERSION), methods=['GET'])
    def get_example():
        num_examples = int(request.args.get('numExamples'))
        return json.dumps({
            'numExamples': num_examples,
            'data': [examples[i] for i in random.sample(range(len(examples)), k=num_examples)]
        })

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != '' and os.path.exists(params.dist_dir + path):
            return send_from_directory(params.dist_dir, path)
        else:
            return send_from_directory(params.dist_dir, 'index.html')

    return app


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.DEFAULTS)
    flags = config.model_config(defaults).FLAGS
    params = util.load_config(flags, util.config_path(flags))  # Loads a pre-existing config otherwise == params
    app = demo(config.gpu_config(), params)
    app.run(port=5000)
