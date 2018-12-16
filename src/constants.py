import os


class Urls:
    """ Constant URLs for downloading data.
        The following keys are defined:
        * TRAIN_SQUAD_1: URL to train set of Squad 1.
        * DEV_SQUAD_1: URL to dev set of Squad 1.
        * GLOVE_380_300D_URL: URL to 300D cased GlOvE vectors.
    """
    # Download URLs
    TRAIN_SQUAD_1 = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json'
    DEV_SQUAD_1 = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'
    TRAIN_SQUAD_2 = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json'
    DEV_SQUAD_2 = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'
    GLOVE_380_300D_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'


class Datasets:
    """ Constant dataset keys.
        The following keys are defined:
        * SQUAD_1: Squad v1.0 - v1.1.
        * SQUAD_2: Squad v2.0.
    """
    SQUAD_1 = 'squad_v1'
    SQUAD_2 = 'squad_v2'


class FilePaths:
    """ Constant filepaths for saving/loading data.
        The following keys are defined:
        * DEFAULTS: Path to the default model parameters.
    """
    DEFAULTS = os.path.abspath('./data/defaults.json')


class FileNames:
    """ Constant URLs for downloading data.
        The following keys are defined:
        * DEFAULTS: Path to the default model parameters.
        * DEFAULTS: Path to the default model parameters.
        * DEFAULTS: Path to the default model parameters.
    """
    TRAIN_SQUAD_1 = 'train-v1.1.json'
    DEV_SQUAD_1 = 'dev-v1.1.json'
    TRAIN_SQUAD_2 = 'train-v2.0.json'
    DEV_SQUAD_2 = 'dev-v2.0.json'
    TRAIN_DEFAULT = 'train.json'
    DEV_DEFAULT = 'dev.json'
    EXAMPLES = 'examples.json'
    INDEX = '{embedding_type}_index.json'
    EMBEDDINGS = '{embedding_type}_embeddings.npy'
    CONTEXT = '{data_type}_contexts.json'
    ANSWERS = '{data_type}_answers.json'
    TF_RECORD = '{name}.tfrecord'
    TRAIN = 'train'
    DEV = 'dev'


class DirNames:
    """ Constant URLs for downloading data.
        The following keys are defined:
        * DEFAULTS: Path to the default model parameters.
        * DEFAULTS: Path to the default model parameters.
        * DEFAULTS: Path to the default model parameters.
    """
    CHECKPOINTS = 'checkpoints'
    LOGS = 'logs'
    RECORDS = 'records'
    PROCESSED = 'processed'
    EMBEDDINGS = 'embeddings'
    SQUAD_1 = Datasets.SQUAD_1
    SQUAD_2 = Datasets.SQUAD_2


class Modes:
    """ Standard names for repo modes.
        The following keys are defined:
        * `TRAIN`: training mode.
        * `TEST`: testing mode.
        * `PREPROCESS`: preprocess mode.
        * `DEMO`: inference mode.
        * `DOWNLOAD`: download mode.
    """
    DEMO = 'demo'
    DOWNLOAD = 'download'
    PREPROCESS = 'preprocess'
    TEST = 'test'
    TRAIN = 'train'


class EmbeddingTypes:
    """ Names for embedding types.
        The following keys are defined:
        * `WORD`: Word embeddings name.
        * `TRAINABLE`: Trainable embeddings name.
        * `CHAR`: Character embeddings name.
    """
    WORD = 'word'
    TRAINABLE = 'trainable'
    CHAR = 'char'

    @staticmethod
    def get_list():
        """ Returns a list of all supported     embedding types """
        return [EmbeddingTypes.WORD, EmbeddingTypes.TRAINABLE, EmbeddingTypes.CHAR]
