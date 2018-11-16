from src import config, constants, util
from train import train
from preprocess import preprocess
from test import test
from demo import demo


def main(config, flags):
    hparams = flags.FLAGS
    mode = hparams.mode.lower()

    if mode == 'train':
        train(config, hparams)
    elif mode == 'preprocess':
        preprocess(hparams)
    elif mode == 'test':
        test(config, hparams)
    elif mode == 'demo':
        app = demo(config, hparams)
        app.run(port=5000)
    else:
        print('Unknown Mode.')
        exit(0)


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.defaults.value)
    main(config.gpu_config(), config.model_config(defaults))
