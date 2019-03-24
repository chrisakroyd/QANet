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
    EMBEDDING_URL = GLOVE_380_300D_URL


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
    """ Variety of constant filenames.
        The following keys are defined:
        * TRAIN_SQUAD_1: Name of the raw Squad 1 train file.
        * DEV_SQUAD_1: Name of the raw Squad 1 dev file.
        * TRAIN_SQUAD_2: Name of the raw Squad 2 train file.
        * DEV_SQUAD_2: Name of the raw Squad 2 dev file.
        * TRAIN_DEFAULT: Default filename for an unrecognised train dataset.
        * DEV_DEFAULT: Default filename for an unrecognised dev dataset.
        * EXAMPLES: Filename to store examples of data.
        * INDEX: Filename + type for word/character index files.
        * EMBEDDINGS: Filename + type for word/character embedding files.
        * CONTEXT: Filename + type for storing context related information for eval/test files.
        * ANSWERS: Filename + type for storing answer related information for eval/test files.
        * TF_RECORD: Tfrecord template string for storing processed train/dev files.
        * TRAIN: String representing train mode data.
        * DEV: String representing val mode data.
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
    CONFIG = 'model_config.json'
    RESULTS = '{timestamp}_results.json'
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'


class DirNames:
    """ Constant directory names for storing data/logs/checkpoints.
        The following keys are defined:
        * CHECKPOINTS: Name of the directory to store checkpoint files.
        * LOGS: Name of the directory to store log files.
        * RECORDS: Name of the directory to store .tfrecord files.
        * PROCESSED: Name of the directory to store processed data.
        * EMBEDDINGS: Name of the directory to store raw embeddings.
        * SQUAD_1: Name of the squad v1 directory.
        * SQUAD_2: Name of the squad v2 directory.
    """
    CHECKPOINTS = 'checkpoints'
    LOGS = 'logs'
    PROCESSED = 'processed'
    TRANSLATED = 'translated_{data}_{lang}'
    EMBEDDINGS = 'embeddings'
    SQUAD_1 = Datasets.SQUAD_1
    SQUAD_2 = Datasets.SQUAD_2


class Modes:
    """ Standard names for repo modes.
        The following keys are defined:
        * TRAIN: training mode.
        * TEST: testing mode.
        * PREPROCESS: preprocess mode.
        * DEBUG: Debug mode.
        * DEMO: inference mode.
        * DOWNLOAD: download mode.
    """
    DEBUG = 'debug'
    DEMO = 'demo'
    DOWNLOAD = 'download'
    PREPROCESS = 'preprocess'
    TEST = 'test'
    TRAIN = 'train'
    TRANSLATE = 'translate'


class EmbeddingTypes:
    """ Names for embedding types.
        The following keys are defined:
        * WORD: Word embeddings name.
        * TRAINABLE: Trainable embeddings name.
        * CHAR: Character embeddings name.
    """
    WORD = 'word'
    TRAINABLE = 'trainable'
    CHAR = 'char'

    @staticmethod
    def as_list():
        """ Returns a list of all supported embedding types """
        return [EmbeddingTypes.WORD, EmbeddingTypes.TRAINABLE, EmbeddingTypes.CHAR]


class ErrorMessages:
    """ Constant error messages.
        The following keys are defined:
        * NO_CONTEXT: Key for context missing.
        * NO_QUERY: Key for QUERY missing.
        * INVALID_CONTEXT: Context field is invalid.
        * INVALID_QUERY: Query field is invalid.
        * OUT_OF_RANGE_ERR: Internal error related to iterators running out of data.
    """
    NO_CONTEXT = 'Context key missing from body of POST request.'
    NO_QUERY = 'Query key missing from body of POST request.'
    INVALID_CONTEXT = 'Context must be longer than 0 excluding space characters.'
    INVALID_QUERY = 'Query must be longer than 0 excluding space characters.'
    OUT_OF_RANGE_ERR = 'Iterator out of range, attempted to call too many times. (Please report this error)'


class PlaceholderKeys:
    """ Constants for placeholder dict keys.
        INPUT_KEYS: Keys for model input, e.g. words, characters.
        LABEL_KEYS: Keys for labels.
        ID_KEY: Answer id key.
    """
    INPUT_KEYS = ['context_words', 'context_chars', 'context_elmo', 'context_length',
                  'query_words', 'query_chars', 'query_elmo', 'query_length']
    LABEL_KEYS = ['answer_starts', 'answer_ends', 'answer_id', 'is_impossible']
    ID_KEY = ['answer_id']
