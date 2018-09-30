import tensorflow as tf
# import flask
from src.constants import FilePaths
from src.util import namespace_json
from src.config import gpu_config, model_config


def demo(config, flags):
    return


if __name__ == '__main__':
    defaults = namespace_json(path=FilePaths.defaults.value)
    tf.app.run(main=demo, argv=[gpu_config(), model_config(defaults)])
