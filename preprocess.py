from src.constants import FilePaths
from src.util import namespace_json
from src.config import model_config


def preprocess(config):
    return


if __name__ == '__main__':
    defaults = namespace_json(path=FilePaths.defaults.value)
    preprocess(model_config(defaults))
