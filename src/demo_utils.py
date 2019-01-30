import math
import numpy as np
from src import util


def split_text(context_tokens, max_tokens=400):
    """ Splits a long sample of text into n samples smaller than max tokens.
        Args:
            context_tokens:
            max_tokens: Maximum number of tokens in a segment.
        Returns:
            A list of texts smaller than or equal to max_tokens.
    """
    num_splits = math.ceil(len(context_tokens) / max_tokens)
    context_tokens = np.array(context_tokens, dtype=np.str)
    split_tokens = np.array_split(context_tokens, num_splits)
    split_tokens = [tokens.tolist() for tokens in split_tokens]
    split_lengths = [len(tokens) for tokens in split_tokens]
    split_tokens = util.pad_to_max_length(split_tokens, split_lengths)
    return split_tokens, split_lengths


def get_params(data, keys={'context', 'query'}):
    """ Creates dict of params from request restricted to a set of allowed keys."""
    return {key: data[key] for key in keys if key in data}


def get_answer_texts(context_tokens, answer_starts, answer_ends):
    """
        Given a context, answer starts and answer end positions, extracts the original text for each answer pair.
        Args:
            context_tokens: Tokenized form of the original context.
            answer_starts: An answer start pointer.
            answer_ends: An answer end pointer.
        Returns:
            A list of answer texts.
    """
    answer_texts = []

    for tokens, starts, ends in zip(context_tokens, answer_starts, answer_ends):
        answer_text = tokens[starts:ends + 1]
        answer_text = ' '.join(answer_text)
        answer_texts.append(answer_text)
    return answer_texts


def get_predict_response(context_tokens, query_tokens, answer_starts, answer_ends, p_starts, p_ends, orig_body):
    """
        Generates a formatted response message containing successful predictions as well as original parameters.
        Args:
            context_tokens: Tokenized form of the original context.
            query_tokens: Tokenized form of the original query.
            answer_starts: The predicted answer start position.
            answer_ends: The predicted answer end position.
            p_starts: The softmax probabilities for start position over all context tokens.
            p_ends: The softmax probabilities for end position over all context tokens.
            orig_body: Original parameters send with the prediction request.
        Returns:
            Formatted prediction success response message.
    """
    answer_starts = answer_starts.tolist()
    answer_ends = answer_ends.tolist()
    p_starts = p_starts.tolist()
    p_ends = p_ends.tolist()
    answer_texts = get_answer_texts(context_tokens, answer_starts, answer_ends)

    response_iter = zip(context_tokens, query_tokens, answer_texts, answer_starts, answer_ends, p_starts, p_ends)
    data = []

    for c_tokens, q_tokens, ans_text, ans_start, ans_end, p_start, p_end in response_iter:
        data.append({
            'contextTokens': c_tokens,
            'queryTokens': q_tokens,
            'answerText': ans_text,
            'answerStart': ans_start,
            'answerEnd': ans_end,
            'startProb': p_start,
            'endProb': p_end,
            'answerProb':  p_start[ans_start] * p_end[ans_end]
        })

    return {
        'numPredictions': len(p_starts),
        'data': data,
        'parameters': get_params(orig_body)
    }


def get_error_response(error_message, orig_body, error_code=0):
    """
        Generates a formatted error response with the given error message and error code.
        Args:
            error_code: A unique id for this error.
            orig_body: The original parameters sent with the request which were invalid.
            error_message: Cause of this error.
        Returns:
            Error dict for the response.
    """
    return {
        'errorCode': error_code,
        'errorMessage': error_message,
        'parameters': get_params(orig_body)
    }
