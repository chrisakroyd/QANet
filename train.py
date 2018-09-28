import tensorflow as tf
from src.constants import FilePaths
from src.util import namespace_json
from src.config import gpu_config, model_config


def train(_, config):
    return


if __name__ == '__main__':
    defaults = namespace_json(path=FilePaths.defaults)
    tf.app.run(main=train, argv=[gpu_config(), model_config(defaults)])
