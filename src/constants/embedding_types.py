from enum import Enum


class EmbeddingTypes(Enum):
    """
        Enum for constant values relating to saving/loading embedding files.
    """
    WORD = 'word'
    TRAINABLE = 'trainable'
    CHAR = 'char'
