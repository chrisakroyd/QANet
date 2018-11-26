from src import config, constants, util
from train import train
from preprocess import preprocess
from test import test
from demo import demo


def main(config, flags):
    params = flags.FLAGS
    mode = params.mode.lower()

    if mode == 'train':
        train(config, params)
    elif mode == 'preprocess':
        preprocess(params)
    elif mode == 'test':
        test(config, params)
    elif mode == 'demo':
        app = demo(config, params)
        app.run(port=params.demo_server_port)
    else:
        print('Unknown Mode.')
        exit(0)


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.defaults.value)
    main(config.gpu_config(), config.model_config(defaults))
