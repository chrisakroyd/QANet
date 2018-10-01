import os
from src.constants import FileNames


def get_directories(hparams):
    squad_dir = os.path.abspath(hparams.squad_dir)
    data_dir = os.path.abspath(hparams.data_dir)
    out_dir = os.path.abspath(hparams.out_dir)
    model_dir = os.path.join(out_dir, 'checkpoints')
    logs_dir = os.path.join(out_dir, 'logs')
    return squad_dir, data_dir, model_dir, logs_dir


def raw_data_paths(hparams):
    squad_dir, _, _, _ = get_directories(hparams)
    # Where we find the data
    train_path = os.path.join(squad_dir, FileNames.train_squad_1.value)
    dev_path = os.path.join(squad_dir, FileNames.dev_squad_1.value)

    return train_path, dev_path


def processed_data_paths(hparams):
    _, data_dir, _, _ = get_directories(hparams)
    paths = []
    for data_type in FileNames.data_types.value:
        paths += [
            os.path.join(data_dir, FileNames.context.value.format(data_type)),
            os.path.join(data_dir, FileNames.answers.value.format(data_type)),
        ]
    return paths


def embedding_paths(hparams):
    _, data_dir, _, _ = get_directories(hparams)
    paths = []
    for embedding_type in FileNames.embedding_types.value:
        paths += [
            os.path.join(data_dir, FileNames.index.value.format(embedding_type)),
            os.path.join(data_dir, FileNames.embeddings.value.format(embedding_type))
        ]
    return paths


def train_paths(hparams):
    _, data_dir, model_dir, logs_dir = get_directories(hparams)

    model_path = os.path.join(model_dir, hparams.run_name)
    logs_path = os.path.join(logs_dir, hparams.run_name)

    return data_dir, model_path, logs_path
