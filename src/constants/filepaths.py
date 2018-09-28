from enum import Enum


class FilePaths(Enum):
    defaults = ''


class FileNames(Enum):
    train_squad_1 = 'train-v.1.1.json'
    dev_squad_1 = 'dev-v.1.1.json'
    word_index = ''
    trainable_index = ''
    char_index = ''
    word_embeddings = ''
    trainable_embeddings = ''
    char_embeddings = ''
    indexed_name = ''
    context_name = ''
    elmo_name = ''
    answers_name = ''
