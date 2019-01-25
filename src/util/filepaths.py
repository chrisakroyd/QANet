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
    processed_data_dir = os.path.join(data_dir, constants.DirNames.PROCESSED, params.dataset)
    models_dir = os.path.abspath(params.models_dir)

    return data_dir, processed_data_dir, models_dir


def raw_data_directory(params, dataset=None):
    """ Returns a path to the directory where the data for the given dataset is stored. """
    if dataset is None:
        dataset = params.dataset

    data_dir, _, _ = get_directories(params)
    raw_data_dir = os.path.join(data_dir, constants.DirNames.RAW, dataset)
    return raw_data_dir


def get_filenames(dataset):
    """ Gets the filenames for a specific dataset, if dataset isn't listed returns defaults.
        Args:
            dataset: A string dataset key.
        returns:
            String paths for train + dev sets.
    """
    if dataset == constants.Datasets.SQUAD_1:
        return constants.FileNames.TRAIN_SQUAD_1, constants.FileNames.DEV_SQUAD_1
    elif dataset == constants.Datasets.SQUAD_2:
        return constants.FileNames.TRAIN_SQUAD_2, constants.FileNames.DEV_SQUAD_2
    else:
        return constants.FileNames.TRAIN_DEFAULT, constants.FileNames.DEV_DEFAULT


def raw_data_paths(params, dataset=None):
    """ Generates paths to raw data.
        Args:
            params: A dictionary of parameters.
            dataset: String dataset key.
        returns:
            String paths for raw squad train + dev sets.
    """
    if dataset is None:
        dataset = params.dataset

    raw_data_dir = raw_data_directory(params, dataset)
    train_name, dev_name = get_filenames(dataset)
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
    _, processed_dir, _ = get_directories(params)
    train, dev, test = constants.FileNames.TRAIN, constants.FileNames.DEV, constants.FileNames.TEST

    paths = (
        os.path.join(processed_dir, constants.FileNames.CONTEXT.format(data_type=train)),
        os.path.join(processed_dir, constants.FileNames.ANSWERS.format(data_type=train)),
        os.path.join(processed_dir, constants.FileNames.CONTEXT.format(data_type=dev)),
        os.path.join(processed_dir, constants.FileNames.ANSWERS.format(data_type=dev)),
        os.path.join(processed_dir, constants.FileNames.CONTEXT.format(data_type=test)),
        os.path.join(processed_dir, constants.FileNames.ANSWERS.format(data_type=test)),
    )

    return paths


def index_paths(params):
    """ Generates paths to word indexes.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for loading word, character and trainable indexes.
    """
    _, processed_dir, _ = get_directories(params)

    paths = []

    for embed_type in constants.EmbeddingTypes.as_list():
        paths += [os.path.join(processed_dir, constants.FileNames.INDEX.format(embedding_type=embed_type))]

    return paths


def embedding_paths(params):
    """ Generates paths to saved embedding files.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for loading word, character and trainable embeddings.
    """
    _, processed_dir, _ = get_directories(params)
    paths = []
    for embed_type in constants.EmbeddingTypes.as_list():
        paths += [
            os.path.join(processed_dir, constants.FileNames.EMBEDDINGS.format(embedding_type=embed_type))
        ]
    return paths


def train_paths(params):
    """ Generates paths to save trained models and logs for each run.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for loading data, saved models and saved logs.
    """
    data_dir, _, out_dir = get_directories(params)

    model_dir = os.path.join(out_dir, constants.DirNames.CHECKPOINTS)
    logs_dir = os.path.join(out_dir, constants.DirNames.LOGS)

    model_path = os.path.join(out_dir, model_dir, params.run_name)
    logs_path = os.path.join(out_dir, logs_dir, params.run_name)

    return out_dir, model_path, logs_path


def results_path(params):
    _, _, logs_path = train_paths(params)
    return os.path.join(logs_path, 'results.json')


def tf_record_paths(params):
    """ Generates a paths to .tfrecord files for train, dev and test.
        Args:
            params: A dictionary of parameters.
        returns:
            A string path to .tfrecord file.
    """
    _, processed_data_dir, _ = get_directories(params)

    paths = (
        os.path.join(processed_data_dir, constants.FileNames.TF_RECORD.format(name=constants.FileNames.TRAIN)),
        os.path.join(processed_data_dir, constants.FileNames.TF_RECORD.format(name=constants.FileNames.DEV)),
        os.path.join(processed_data_dir, constants.FileNames.TF_RECORD.format(name=constants.FileNames.TEST))
    )

    return paths


def examples_path(params):
    _, processed_dir, _ = get_directories(params)
    return os.path.join(processed_dir, constants.FileNames.EXAMPLES)
