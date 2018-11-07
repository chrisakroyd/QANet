import os
from src import constants


def get_directories(params):
    """ Generates directory paths.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for directories containing unprocessed, processed, models and logs.
    """
    squad_dir = os.path.abspath(params.squad_dir)
    data_dir = os.path.abspath(params.data_dir)
    out_dir = os.path.abspath(params.out_dir)
    model_dir = os.path.join(out_dir, 'checkpoints')
    logs_dir = os.path.join(out_dir, 'logs')
    tf_record_dir = os.path.join(data_dir, 'records')
    return squad_dir, data_dir, out_dir, model_dir, logs_dir, tf_record_dir


def raw_data_paths(params):
    """ Generates paths to raw data.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for raw squad train + dev sets.
    """
    squad_dir, _, _, _, _, _ = get_directories(params)
    # Where we find the data
    train_path = os.path.join(squad_dir, constants.FileNames.TRAIN_SQUAD_1.value)
    dev_path = os.path.join(squad_dir, constants.FileNames.DEV_SQUAD_1.value)

    return train_path, dev_path


def processed_data_paths(params):
    """ Generates paths to processed data.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for processed answers and contexts for train and dev sets.
    """
    _, data_dir, _, _, _, _ = get_directories(params)
    train, dev = constants.Modes.TRAIN.value, constants.Modes.DEV.value
    paths = (
        os.path.join(data_dir, constants.FileNames.CONTEXT.value.format(train)),
        os.path.join(data_dir, constants.FileNames.ANSWERS.value.format(train)),
        os.path.join(data_dir, constants.FileNames.CONTEXT.value.format(dev)),
        os.path.join(data_dir, constants.FileNames.ANSWERS.value.format(dev)),
    )
    return paths


def index_paths(params):
    """ Generates paths to word indexes.
            Args:
                params: A dictionary of parameters.
        returns:
            String paths for loading word, character and trainable indexes.
    """
    _, data_dir, _, _, _, _ = get_directories(params)
    paths = []
    for _, embed_type in constants.EmbeddingTypes.__members__.items():
        paths += [os.path.join(data_dir, constants.FileNames.INDEX.value.format(embed_type.value))]
    return paths


def embedding_paths(params):
    """ Generates paths to saved embedding files.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for loading word, character and trainable embeddings.
    """
    _, data_dir, _, _, _, _ = get_directories(params)
    paths = []
    for _, embed_type in constants.EmbeddingTypes.__members__.items():
        paths += [
            os.path.join(data_dir, constants.FileNames.EMBEDDINGS.value.format(embed_type.value))
        ]
    return paths


def train_paths(params):
    """ Generates paths to save trained models and logs for each run.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for loading data, saved models and saved logs.
    """
    _, data_dir, out_dir, model_dir, logs_dir, _ = get_directories(params)

    model_path = os.path.join(out_dir, model_dir, params.run_name)
    logs_path = os.path.join(out_dir, logs_dir, params.run_name)

    return data_dir, out_dir, model_path, logs_path


def tf_record_paths(params, training):
    """ Generates a path to a .tfrecord file.
        Args:
            params: A dictionary of parameters.
            training: Boolean value for whether we are training or not.
        returns:
            A string path to .tfrecord file.
    """
    _, _, _, _, _, tf_record_dir = get_directories(params)
    if training:
        name = constants.Modes.TRAIN.value
    else:
        name = constants.Modes.DEV.value

    paths = os.path.join(tf_record_dir, '{name}.tfrecord'.format(name=name))

    return paths
