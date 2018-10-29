import os
from src import constants


def get_directories(hparams):
    squad_dir = os.path.abspath(hparams.squad_dir)
    data_dir = os.path.abspath(hparams.data_dir)
    out_dir = os.path.abspath(hparams.out_dir)
    model_dir = os.path.join(out_dir, 'checkpoints')
    logs_dir = os.path.join(out_dir, 'logs')
    return squad_dir, data_dir, out_dir, model_dir, logs_dir


def raw_data_paths(hparams):
    squad_dir, _, _, _, _ = get_directories(hparams)
    # Where we find the data
    train_path = os.path.join(squad_dir, constants.FileNames.TRAIN_SQUAD_1.value)
    dev_path = os.path.join(squad_dir, constants.FileNames.DEV_SQUAD_1.value)

    return train_path, dev_path


def processed_data_paths(hparams):
    _, data_dir, _, _, _ = get_directories(hparams)
    train, dev = constants.Modes.TRAIN.value, constants.Modes.DEV.value
    paths = (
        os.path.join(data_dir, constants.FileNames.CONTEXT.value.format(train)),
        os.path.join(data_dir, constants.FileNames.ANSWERS.value.format(train)),
        os.path.join(data_dir, constants.FileNames.CONTEXT.value.format(dev)),
        os.path.join(data_dir, constants.FileNames.ANSWERS.value.format(dev)),
    )
    return paths


def index_paths(hparams):
    _, data_dir, _, _, _ = get_directories(hparams)
    paths = []
    for _, embed_type in constants.EmbeddingTypes.__members__.items():
        paths += [os.path.join(data_dir, constants.FileNames.INDEX.value.format(embed_type.value))]
    return paths


def embedding_paths(hparams):
    _, data_dir, _, _, _ = get_directories(hparams)
    paths = []
    for _, embed_type in constants.EmbeddingTypes.__members__.items():
        paths += [
            os.path.join(data_dir, constants.FileNames.EMBEDDINGS.value.format(embed_type.value))
        ]
    return paths


def train_paths(hparams):
    _, data_dir, out_dir, model_dir, logs_dir = get_directories(hparams)

    model_path = os.path.join(out_dir, model_dir, hparams.run_name)
    logs_path = os.path.join(out_dir, logs_dir, hparams.run_name)

    return data_dir, out_dir, model_path, logs_path


def tf_record_paths(hparams, train):
    _, data_dir, _, _, _ = get_directories(hparams)
    out_dir = os.path.join(data_dir, 'records')
    if train:
        name = constants.Modes.TRAIN.value
    else:
        name = constants.Modes.DEV.value

    paths = os.path.join(out_dir, '{name}_{shards}.tfrecord'.format(
            name=name, shards=str(hparams.num_shards).zfill(4)))

    return paths
