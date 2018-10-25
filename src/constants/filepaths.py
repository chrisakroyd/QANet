import os
from enum import Enum


class FilePaths(Enum):
    defaults = os.path.abspath('./data/defaults.json')

    def __str__(self):
        return str(self.value)


class FileNames(Enum):
    TRAIN_SQUAD_1 = 'train-v1.1.json'
    DEV_SQUAD_1 = 'dev-v1.1.json'
    INDEX = '{}_index.json'
    EMBEDDINGS = '{}_embeddings.npy'
    CONTEXT = '{}_contexts.json'
    ANSWERS = '{}_answers.json'
