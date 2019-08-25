from src import config, constants, preprocessing, util


def preprocess(params):
    dataset = params.dataset.lower().strip()
    util.make_dirs(util.get_directories(params.data_dir, params.dataset, params.models_dir) +
                   (util.raw_data_directory(params.raw_data_dir, params.dataset), ))

    output_dir = util.processed_data_directory(params.data_dir, params.dataset)

    # Prompt user for confirmation if this action overwrites existing data.
    if util.directory_exists(output_dir) and not util.directory_is_empty(output_dir):
        if not util.yes_no_prompt(constants.Prompts.DATA_EXISTS):
            exit(0)

    if dataset == constants.Datasets.SQUAD_1:
        preprocessing.squad_process(params)
    else:
        raise NotImplementedError('Unsupported dataset: Valid datasets are {}.'.format('squad'))


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.DEFAULTS)
    preprocess(config.model_config(defaults))
