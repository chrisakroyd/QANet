"""
File contains functions related to automatic filepath generation and handling, although this may seem overkill,
it's a big time-saver during development.
"""
import os
import time
from src import constants


def processed_data_directory(data_dir, dataset):
    """ Generates a unique path to save processed data for a dataset """
    processed_data_dir = os.path.join(data_dir, constants.DirNames.PROCESSED, dataset)
    return processed_data_dir


def get_directories(data_dir, dataset, models_dir):
    """ Generates directory paths for data, processed data and saving models """
    return data_dir, processed_data_directory(data_dir, dataset), models_dir


def raw_data_directory(raw_data_dir, dataset):
    """ Returns a path to the directory where the data for the given dataset is stored. """
    raw_dataset = os.path.join(os.path.abspath(raw_data_dir), dataset)
    return raw_dataset


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


def raw_data_paths(raw_data_dir, dataset):
    """ Generates paths to raw data.
        Args:
            raw_data_dir: Directory where we store raw data.
            dataset: String dataset key.
        returns:
            String paths for raw squad train + dev sets.
    """
    raw_dataset = raw_data_directory(raw_data_dir, dataset)
    train_name, dev_name = get_filenames(dataset)
    # Where we find the data
    train_path = os.path.join(raw_dataset, train_name)
    dev_path = os.path.join(raw_dataset, dev_name)

    return train_path, dev_path


def processed_data_paths(data_dir, dataset):
    """ Generates paths to processed data.
        Args:
            data_dir: Directory in which we wish to save processed data.
            dataset: Name of the dataset we want to generate paths to.
        returns:
            String paths for processed answers and contexts for train and dev sets.
    """
    processed_dir = processed_data_directory(data_dir, dataset)
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


def index_paths(data_dir, dataset):
    """ Generates paths to word indexes.
        Args:
            data_dir: Directory in which we wish to save processed data.
            dataset: Name of the dataset we want to generate paths to.
        returns:
            String paths for loading word, character and trainable indexes.
    """
    processed_dir = processed_data_directory(data_dir, dataset)

    paths = []

    for embed_type in constants.EmbeddingTypes.as_list():
        paths += [os.path.join(processed_dir, constants.FileNames.INDEX.format(embedding_type=embed_type))]

    return paths


def embedding_paths(data_dir, dataset):
    """ Generates paths to saved embedding files.
        Args:
            data_dir: Directory in which we wish to save processed data.
            dataset: Name of the dataset we want to generate paths to.
        returns:
            String paths for loading word, character and trainable embeddings.
    """
    processed_dir = processed_data_directory(data_dir, dataset)
    paths = []
    for embed_type in constants.EmbeddingTypes.as_list():
        paths += [
            os.path.join(processed_dir, constants.FileNames.EMBEDDINGS.format(embedding_type=embed_type))
        ]
    return paths


def save_paths(models_dir, run_name):
    """ Generates paths to save trained models and logs for each run.
        Args:
            models_dir: Directory in which we save checkpoints/logs.
            run_name: Name of the run which we want to generate paths to.
        returns:
            String paths for loading data, saved models and saved logs.
    """
    model_path = os.path.join(models_dir, constants.DirNames.CHECKPOINTS, run_name)
    logs_path = os.path.join(models_dir, constants.DirNames.LOGS, run_name)
    return model_path, logs_path


def results_path(models_dir, run_name):
    """ Generates a path to save test results. """
    _, logs_path = save_paths(models_dir, run_name)
    return os.path.join(logs_path, constants.FileNames.RESULTS.format(timestamp=int(time.time())))


def config_path(models_dir, run_name):
    """ Generates a path to a .json file containing parameters used for a train run. """
    model_path, _ = save_paths(models_dir, run_name)
    return os.path.join(model_path, constants.FileNames.CONFIG)


def tf_record_paths(data_dir, dataset):
    """ Generates a paths to .tfrecord files for train, dev and test.
        Args:
            data_dir: Directory in which we wish to save processed data.
            dataset: Name of the dataset we want to generate paths to.
        returns:
            A string path to .tfrecord file.
    """
    processed_dir = processed_data_directory(data_dir, dataset)

    paths = (
        os.path.join(processed_dir, constants.FileNames.TF_RECORD.format(name=constants.FileNames.TRAIN)),
        os.path.join(processed_dir, constants.FileNames.TF_RECORD.format(name=constants.FileNames.DEV)),
        os.path.join(processed_dir, constants.FileNames.TF_RECORD.format(name=constants.FileNames.TEST))
    )

    return paths


def examples_path(data_dir, dataset):
    processed_dir = processed_data_directory(data_dir, dataset)
    return os.path.join(processed_dir, constants.FileNames.EXAMPLES)
