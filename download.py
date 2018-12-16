import os
from src import config, constants, util


def download(params):
    directories = util.get_directories(params)
    util.make_dirs(directories)
    train_path, dev_path = util.raw_data_paths(params)

    if not os.path.exists(train_path) and not os.path.exists(dev_path):
        print('Downloading Squad v1.1...')
        util.download_json(constants.Urls.TRAIN_SQUAD_1, train_path)
        util.download_json(constants.Urls.DEV_SQUAD_1, dev_path)
    else:
        print('Squad v1.1 data already exists, skipping...')

    # If an embedding file doesn't exist at the specified location, download them.
    if not os.path.exists(params.embeddings_path):
        print('Downloading embeddings, this may take some time...')
        _, data_dir, _, _, _ = directories
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
