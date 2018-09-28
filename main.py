import tensorflow as tf
from src.constants import FilePaths
from src.util import namespace_json
from src.config import gpu_config, model_config
from train import train
from preprocess import preprocess
from test import test
from demo import demo


def main(config, flags):
    hparams = flags.FLAGS
    mode = config.mode.lower()

    if mode == 'train':
        train(config, hparams)
    elif mode == 'preprocess':
        preprocess(hparams)
    elif mode == 'test':
        test(config, hparams)
    elif mode == 'demo':
        demo(config, hparams)
    else:
        print('Unknown Mode.')
        exit(0)


if __name__ == '__main__':
    defaults = namespace_json(path=FilePaths.defaults)
    tf.app.run(main=main, argv=[gpu_config(), model_config(defaults)])
