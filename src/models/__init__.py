from .qanet import QANet
from .qanet_ut import UTQANet
from . qanet_contextual import QANetContextual
from src import constants


def create_model(word_matrix, character_matrix, trainable_matrix, params):
    """
        Utility function that creates a given model from its string handle (Found in src/constants.py).

        Args:
            word_matrix: n-dim Matrix for words.
            character_matrix: n-dim Matrix for trainable characters.
            trainable_matrix: n-dim Matrix for trainable words.
            params: Tf.flag parameter object.
        Returns:
            Initialised Keras Model.
    """
    if params.model == constants.ModelTypes.QANET:
        model = QANet(word_matrix, character_matrix, trainable_matrix, params)
    elif params.model == constants.ModelTypes.QANET_CONTEXTUAL:
        model = QANetContextual(word_matrix, character_matrix, trainable_matrix, params)
    elif params.model == constants.ModelTypes.UNIVERSAL_TRANSFORMER:
        model = UTQANet(word_matrix, character_matrix, trainable_matrix, params)
    else:
        raise ValueError('Unsupported model type.')

    return model
