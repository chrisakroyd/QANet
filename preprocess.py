from src import config, constants, preprocessing, util


def preprocess(params):
    dataset = params.dataset.lower().strip()
    util.make_dirs(util.get_directories(params))

    if dataset == constants.Datasets.SQUAD_1:
        preprocessing.squad_process(params)
    else:
        raise NotImplementedError('Unsupported dataset: Valid datasets are {}.'.format('squad'))


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.DEFAULTS)
    preprocess(config.model_config(defaults))
