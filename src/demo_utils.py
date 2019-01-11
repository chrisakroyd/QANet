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
        answer_text = tokens[starts[0]:ends[0] + 1]
        answer_text = ' '.join(answer_text)
        answer_texts.append(answer_text)
    # TODO: Wrapping lists to prep for multiple contexts -> this will break so 100% needs to be fixed.
    return [answer_texts]


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
    if isinstance(context_tokens[0], str):
        context_tokens = [context_tokens]
    if isinstance(query_tokens[0], str):
        query_tokens = [query_tokens]
    # TODO: Wrapping lists to prep for multiple contexts -> this will break so 100% needs to be fixed before release.
    answer_starts = [answer_starts.tolist()]
    answer_ends = [answer_ends.tolist()]
    p_starts = p_starts.tolist()
    p_ends = p_ends.tolist()
    answer_texts = get_answer_texts(context_tokens, answer_starts, answer_ends)

    resp_iterable = zip(context_tokens, query_tokens, answer_texts, answer_starts, answer_ends, p_starts, p_ends)
    data = []

    for c_tokens, q_tokens, ans_txts, ans_starts, ans_ends, p_start, p_end in resp_iterable:
        data.append({
            'contextTokens': c_tokens,
            'queryTokens': q_tokens,
            'answerText': ans_txts[-1],
            'answerStart': ans_starts[-1],
            'answerEnd': ans_ends[-1],
            'startProb': p_start,
            'endProb': p_end,
        })

    return {
        'numPredictions': len(p_starts),
        # TODO: Revist this to grab the multiplied prob from the prediction matrix (P(start) * P(end)).
        'bestAnswer': answer_texts[0][-1],
        'data': data,
        'parameters': {
            'context': orig_body['context'],
            'query': orig_body['query']
        }
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

    params = {}

    if 'context' in orig_body:
        params['context'] = orig_body['context']

    if 'query' in orig_body:
        params['query'] = orig_body['query']

    return {
        'error_code': error_code,
        'error_message': error_message,
        'parameters': params
    }
