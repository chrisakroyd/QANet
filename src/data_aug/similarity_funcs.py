def char_ngrams(text, n=2):
    """ Generates character ngrams """
    return set([text[i:i + n] for i in range(len(text) - 1)])


def ngram_sim(answer, text, n=2):
    """ char ngram sim w/ dice coefficient """
    answer_ngrams = char_ngrams(answer.lower(), n)
    text_ngrams = char_ngrams(text.lower(), n)

    intersect = answer_ngrams & text_ngrams

    if len(intersect) == 0:
        return 0.0

    sim = 2 * len(intersect) / (len(answer_ngrams) + len(text_ngrams))
    return sim


def jaro_winkler(answer_word, context_word):
    """ Implementation from https://github.com/jamesturk/jellyfish """
    ying = answer_word
    yang = context_word

    ying_len = len(ying)
    yang_len = len(yang)

    if not ying_len or not yang_len:
        return 0.0

    min_len = max(ying_len, yang_len)
    search_range = (min_len // 2) - 1
    if search_range < 0:
        search_range = 0

    ying_flags = [False] * ying_len
    yang_flags = [False] * yang_len

    # looking only within search range, count & flag matched pairs
    common_chars = 0
    for i, ying_ch in enumerate(ying):
        low = i - search_range if i > search_range else 0
        hi = i + search_range if i + search_range < yang_len else yang_len - 1
        for j in range(low, hi + 1):
            if not yang_flags[j] and yang[j] == ying_ch:
                ying_flags[i] = yang_flags[j] = True
                common_chars += 1
                break

    # short circuit if no characters match
    if not common_chars:
        return 0.0

    # count transpositions
    k = trans_count = 0
    for i, ying_f in enumerate(ying_flags):
        if ying_f:
            for j in range(k, yang_len):
                if yang_flags[j]:
                    k = j + 1
                    break
            if ying[i] != yang[j]:
                trans_count += 1
    trans_count /= 2

    # adjust for similarities in nonmatched characters
    common_chars = float(common_chars)
    weight = ((common_chars / ying_len + common_chars / yang_len +
               (common_chars - trans_count) / common_chars)) / 3

    # winkler modification: continue to boost if strings are similar
    if weight > 0.7 and ying_len > 3 and yang_len > 3:
        # adjust for up to first 4 chars in common
        j = min(min_len, 4)
        i = 0
        while i < j and ying[i] == yang[i] and ying[i]:
            i += 1
        if i:
            weight += i * 0.1 * (1.0 - weight)

    return weight
