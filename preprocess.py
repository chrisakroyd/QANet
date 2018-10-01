from src.config import model_config
from src.constants import FilePaths
from src.preprocessing import squad_process
from src.util import namespace_json, get_directories, make_dirs


def preprocess(hparams):
    dataset = hparams.dataset.lower()
    make_dirs(get_directories(hparams))

    if dataset == 'squad':
        squad_process(hparams)
    else:
        raise NotImplementedError('Unsupported dataset: Valid datasets are {}.'.format('squad'))


if __name__ == '__main__':
    defaults = namespace_json(path=FilePaths.defaults.value)
    preprocess(model_config(defaults))
