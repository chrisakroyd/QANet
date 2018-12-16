import os
from src import constants


def get_directories(params):
    """ Generates directory paths.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for directories containing unprocessed, processed, models and logs.
    """
    data_dir = os.path.abspath(params.data_dir)
    raw_data_dir = os.path.join(data_dir, params.dataset)
    processed_data_dir = os.path.join(data_dir, constants.DirNames.PROCESSED, params.dataset)
    models_dir = os.path.abspath(params.models_dir)

    return raw_data_dir, data_dir, processed_data_dir, models_dir


def get_filenames(params):
    if params.dataset == constants.Datasets.SQUAD_1:
        return constants.FileNames.TRAIN_SQUAD_1, constants.FileNames.DEV_SQUAD_1
    elif params.dataset == constants.Datasets.SQUAD_2:
        return constants.FileNames.TRAIN_SQUAD_2, constants.FileNames.DEV_SQUAD_2
    else:
        return constants.FileNames.TRAIN_DEFAULT, constants.FileNames.DEV_DEFAULT


def raw_data_paths(params):
    """ Generates paths to raw data.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for raw squad train + dev sets.
    """
    raw_data_dir, _, _, _ = get_directories(params)
    train_name, dev_name = get_filenames(params)
    # Where we find the data
    train_path = os.path.join(raw_data_dir, train_name)
    dev_path = os.path.join(raw_data_dir, dev_name)

    return train_path, dev_path


def processed_data_paths(params):
    """ Generates paths to processed data.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for processed answers and contexts for train and dev sets.
    """
    _, data_dir, processed_dir, _ = get_directories(params)
    train, dev = constants.FileNames.TRAIN, constants.FileNames.DEV

    paths = (
        os.path.join(data_dir, processed_dir, constants.FileNames.CONTEXT.format(data_type=train)),
        os.path.join(data_dir, processed_dir, constants.FileNames.ANSWERS.format(data_type=train)),
        os.path.join(data_dir, processed_dir, constants.FileNames.CONTEXT.format(data_type=dev)),
        os.path.join(data_dir, processed_dir, constants.FileNames.ANSWERS.format(data_type=dev)),
    )
    return paths


def index_paths(params):
    """ Generates paths to word indexes.
            Args:
                params: A dictionary of parameters.
        returns:
            String paths for loading word, character and trainable indexes.
    """
    _, data_dir, processed_dir, _ = get_directories(params)

    paths = []
    for embed_type in constants.EmbeddingTypes.get_list():
        paths += [os.path.join(data_dir, processed_dir, constants.FileNames.INDEX.format(embedding_type=embed_type))]
    return paths


def embedding_paths(params):
    """ Generates paths to saved embedding files.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for loading word, character and trainable embeddings.
    """
    _, data_dir, processed_dir, _ = get_directories(params)
    paths = []
    for embed_type in constants.EmbeddingTypes.get_list():
        paths += [
            os.path.join(data_dir, processed_dir, constants.FileNames.EMBEDDINGS.format(embedding_type=embed_type))
        ]
    return paths


def train_paths(params):
    """ Generates paths to save trained models and logs for each run.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for loading data, saved models and saved logs.
    """
    _, data_dir, _, out_dir = get_directories(params)

    model_dir = os.path.join(out_dir, constants.DirNames.CHECKPOINTS)
    logs_dir = os.path.join(out_dir, constants.DirNames.LOGS)

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
    _, _, processed_data_dir, _ = get_directories(params)

    tf_record_dir = os.path.join(processed_data_dir, constants.DirNames.RECORDS)

    if training:
        name = constants.FileNames.TRAIN
    else:
        name = constants.FileNames.DEV

    paths = os.path.join(tf_record_dir, constants.FileNames.TF_RECORD.format(name=name))

    return paths


def examples_path(params):
    _, _, processed_dir, _ = get_directories(params)
    return os.path.join(processed_dir, constants.FileNames.EXAMPLES)
