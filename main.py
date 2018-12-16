from src import config, constants, util
from train import train
from preprocess import preprocess
from test import test
from demo import demo
from download import download


def main(config, flags):
    params = flags.FLAGS
    mode = params.mode.lower().strip()

    if mode == constants.Modes.TRAIN:
        train(config, params)
    elif mode == constants.Modes.PREPROCESS:
        preprocess(params)
    elif mode == constants.Modes.TEST:
        test(config, params)
    elif mode == constants.Modes.DEMO:
        app = demo(config, params)
        app.run(port=params.demo_server_port)
    elif mode == constants.Modes.DOWNLOAD:
        download(params)
    else:
        print('Unknown Mode.')
        exit(0)


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.DEFAULTS)
    main(config.gpu_config(), config.model_config(defaults))
