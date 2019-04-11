import tensorflow_hub as hub
from src import constants


def get_hub_module(contextual_model, trainable=True):
    """ Creates a hub module for the given contextual model. """
    contextual_model = contextual_model.lower()

    if contextual_model == constants.ContextualModels.ELMO:
        return hub.Module(constants.Urls.ELMO, trainable=trainable)
    elif contextual_model == constants.ContextualModels.BERT_BASE:
        return hub.Module(constants.Urls.BERT_BASE, trainable=trainable)
    elif contextual_model == constants.ContextualModels.BERT_LARGE:
        return hub.Module(constants.Urls.BERT_LARGE, trainable=trainable)
    else:
        raise NotImplementedError(constants.ErrorMessages.UNSUPPORTED_CONTEXTUAL_MODEL.format(contextual_model))


def get_contextual_dimensionality(contextual_model):
    """ Returns the dimensionality of a contextual embedding model e.g. BERT/ELMo """
    contextual_model = contextual_model.lower()

    if contextual_model == constants.ContextualModels.ELMO:
        return constants.ContextualDimensions.ELMO
    elif contextual_model == constants.ContextualModels.BERT_BASE:
        return constants.ContextualDimensions.BERT_BASE
    elif contextual_model == constants.ContextualModels.BERT_LARGE:
        return constants.ContextualDimensions.BERT_LARGE
    else:
        raise NotImplementedError(constants.ErrorMessages.UNSUPPORTED_CONTEXTUAL_MODEL.format(contextual_model))


def model_support_check(contextual_model):
    """ Raises an error if the contextual model isn't supported or is only partially supported. """
    # TODO: Remove this once contextual models are fully implemented.
    if not contextual_model == constants.ContextualModels.ELMO:
        raise NotImplementedError(constants.ErrorMessages.UNSUPPORTED_CONTEXTUAL_MODEL.format(contextual_model))
