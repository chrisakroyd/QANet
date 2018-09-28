import tensorflow as tf
from src.constants import FilePaths
from src.util import namespace_json
from src.config import model_config


def preprocess(config):
    return


if __name__ == '__main__':
    defaults = namespace_json(path=FilePaths.defaults)
    tf.app.run(main=preprocess, argv=[model_config(defaults)])
