def get_answer_texts(context_tokens, answer_starts, answer_ends):
    """
        Args:
            context_tokens:
            answer_starts:
            answer_ends:
        Returns:

    """
    answer_texts = []

    for tokens, starts, ends in zip(context_tokens, answer_starts, answer_ends):
        answer_text = tokens[starts[0]:ends[0] + 1]
        answer_text = ' '.join(answer_text)
        answer_texts.append(answer_text)
    # TODO: Wrapping lists to prep for multiple contexts -> this will break so 100% needs to be fixed.
    return [answer_texts]


def get_predict_response(context_tokens, query_tokens, answer_starts, answer_ends, p_starts, p_ends):
    """
        Args:
            context_tokens:
            query_tokens:
            answer_starts:
            answer_ends:
            p_starts:
            p_ends:
        Returns:

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
            'answerTexts': ans_txts,
            'answerStarts': ans_starts,
            'answerEnds': ans_ends,
            'startProb': p_start,
            'endProb': p_end,
        })

    return {
        'numPredictions': len(p_starts),
        'data': data,
    }
