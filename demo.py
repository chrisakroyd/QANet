# import flask
from src import config, constants, util


def demo(sess_config, hparams):
    return


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.defaults.value)
    demo(config.gpu_config(), config.model_config(defaults))
