import os
from src import config, constants, util


def download(params):
    data_dir, processed_data_dir, models_dir = util.get_directories(params)
    squad_v1_dir = util.raw_data_directory(params, dataset=constants.Datasets.SQUAD_1)
    squad_v2_dir = util.raw_data_directory(params, dataset=constants.Datasets.SQUAD_2)
    util.make_dirs([data_dir, processed_data_dir, models_dir, squad_v1_dir, squad_v2_dir])
    squad_v1_train_path, squad_v1_dev_path = util.raw_data_paths(params, dataset=constants.Datasets.SQUAD_1)
    squad_v2_train_path, squad_v2_dev_path = util.raw_data_paths(params, dataset=constants.Datasets.SQUAD_2)

    if not util.file_exists(squad_v1_train_path) and not util.file_exists(squad_v1_dev_path):
        print('Downloading Squad v1.1...')
        util.download_json(constants.Urls.TRAIN_SQUAD_1, squad_v1_train_path)
        util.download_json(constants.Urls.DEV_SQUAD_1, squad_v1_dev_path)
    else:
        print('Squad v1.1 data already exists, skipping...')

    if not util.file_exists(squad_v2_train_path) and not util.file_exists(squad_v2_dev_path):
        print('Downloading Squad v2.0...')
        util.download_json(constants.Urls.TRAIN_SQUAD_2, squad_v2_train_path)
        util.download_json(constants.Urls.DEV_SQUAD_2, squad_v2_dev_path)
    else:
        print('Squad v2.0 data already exists, skipping...')

    # If an embedding file doesn't exist at the specified location, download them.
    if not util.file_exists(params.embeddings_path):
        print('Downloading embeddings, this may take some time...')
        embedding_dir = os.path.join(data_dir, constants.DirNames.EMBEDDINGS)
        util.make_dirs(embedding_dir)
        embedding_filename = util.download_unpack_zip(constants.Urls.EMBEDDING_URL, embedding_dir)
        # Update the path to the embeddings file stored in defaults.
        out_embed_path = os.path.join(embedding_dir, embedding_filename)
        default_params = util.load_json(constants.FilePaths.DEFAULTS)
        default_params['embeddings_path'] = out_embed_path
        util.save_json(constants.FilePaths.DEFAULTS, default_params, indent=2)
    else:
        print('Embeddings file already exists, skipping...')


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.DEFAULTS)
    download(config.model_config(defaults).FLAGS)
