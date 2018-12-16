from src import config, constants, util


def download(params):
    util.make_dirs(util.get_directories(params))
    train_path, dev_path = util.raw_data_paths(params)

    util.download_json(constants.Urls.TRAIN_SQUAD_1, train_path)
    util.download_json(constants.Urls.DEV_SQUAD_1, dev_path)
    return


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.DEFAULTS)
    download(config.model_config(defaults).FLAGS)
