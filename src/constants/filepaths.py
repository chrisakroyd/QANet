import os
from enum import Enum


class FilePaths(Enum):
    """
        Enum for direct non-dynamic path strings
    """
    defaults = os.path.abspath('./data/defaults.json')

    def __str__(self):
        return str(self.value)


class FileNames(Enum):
    """
        Enum holding dynamic path strings.
    """
    TRAIN_SQUAD_1 = 'train-v1.1.json'
    DEV_SQUAD_1 = 'dev-v1.1.json'
    EXAMPLES = 'examples.json'
    INDEX = '{}_index.json'
    EMBEDDINGS = '{}_embeddings.npy'
    CONTEXT = '{}_contexts.json'
    ANSWERS = '{}_answers.json'
