import string
import numpy as np
from src import data_aug, preprocessing as prepro

punct = set(string.punctuation)


def top_k_indices(array, k=5):
    """ Gets the indices of the top-k scores from a numpy array. e.g. k=2, array=[0.2, 0.3, 0.1, 0.9], result=[1, 3]"""
    k = min(len(array), k)
    return np.argpartition(array, -k)[-k:]


def get_start_word(answer_tokens):
    """
        First word of some answers are stopwords or punctuation (e.g. the, !). This method finds a word and
        its offset from the start that isn't one or the other.
        Args:
            answer_tokens: list of string tokens
        Returns:
            token: string word
            offset: distance from start of answer.
     """
    for i, token in enumerate(answer_tokens):
        if token not in data_aug.stopwords and token not in punct:
            return token, i
    return answer_tokens[0], 0


def get_end_word(answer_tokens):
    """ Flips the array before getting an end word which isn't a stopword / punctuation. """
    flipped = answer_tokens.copy()  # Take copy so we don't reverse the original array.
    flipped.reverse()
    return get_start_word(flipped)


def compute_similarity_scores(start_word, end_word, tokens, similarity_func=data_aug.jaro_winkler):
    """ Generate a similarity score for each word in the context and the start/end words of the answer text """
    num_comparisons = len(tokens)
    start_sims = np.zeros(shape=(num_comparisons,), dtype=np.float32)
    end_sims = np.zeros(shape=(num_comparisons,), dtype=np.float32)

    for i, word in enumerate(tokens):
        start_sims[i] = similarity_func(word, start_word)
        end_sims[i] = similarity_func(word, end_word)

    return start_sims, end_sims


def most_similar_pointers(start_sims, end_sims, start_offset=0, end_offset=0):
    """ Returns the indices of the most similar start + end words. """
    start_pointer = max(np.argmax(start_sims) - start_offset, 0)
    end_pointer = most_similar_end_pointer(start_pointer, end_sims, end_offset)
    return start_pointer, end_pointer


def most_similar_end_pointer(start_index, end_sims, end_offset=0):
    """ Returns the most similar index after a given start index"""
    end_pointer = np.argmax(end_sims[start_index:]) + start_index + end_offset
    return min(end_pointer, len(end_sims) - 1)


def top_k_indices_after(start_index, array, k=5):
    """ Gets the indices of the top-k scores after an index."""
    return top_k_indices(array[start_index:], k) + start_index


def find_answer(context_text, spans, context_tokens, answer_text, answer_tokens, similarity_func=data_aug.jaro_winkler):
    """
        Given a context, spans and answer_text finds the most similar answer span within the text.

        Implementation is a bit of a mess, but to improve answer quality we use 4 heuristics instead
        of the 1 used in QANet (https://arxiv.org/pdf/1804.09541.pdf) section 3. Also opt to use
        Jaro-winkler vs ngram-sim by default as it leads to better empirical performance.

        Heuristics:
            1. Simply take the best start and end pointers and compare the answer span, if they match -> return.
            2. Take top-k start pointers and find the best end for each, if the text matches -> return.
            3a. Take all pointers from the prev step, extract text + calc sim between extracted text + answer text.
            3b. Take all start pointers, generate top-k end pointers + calc sim between extracted text + answer text.

        Generally heuristics 1 + 2 get the answer accurately so we utilise the slower heuristics 3a + 3b in cases
        where the previous heuristics fail.

        TODO: Code is a mess and needs to be revisited.

        Args:
            context_text: Context in text form, used to extract candidate answers.
            spans: Word character spans, generated during preprocessing.
            context_tokens: Tokenized form of context_text.
            answer_text: The answer text to compare to.
            answer_tokens: Tokenized form of answer_text.
            similarity_func: The similarity function used to score candidate answers.
        Returns:
            Two integer pointers, one to the start word and one to the end word.
    """
    start_word, start_shift = get_start_word(answer_tokens)
    end_word, end_shift = get_end_word(answer_tokens)
    start_sims, end_sims = compute_similarity_scores(start_word, end_word, context_tokens)

    # In the first case we get the two most similar pointers and pray they equal the answer_text
    start_pointer, end_pointer = most_similar_pointers(start_sims, end_sims, start_shift, end_shift)
    candidate_answer = prepro.text_span(context_text, spans, start_pointer, end_pointer)

    if candidate_answer == answer_text:
        return start_pointer, end_pointer

    # Comes into play in the third case where we try and find the most similar span.
    candidate_pointers = [(start_pointer, end_pointer)]

    # In the second case, we take the top-k start pointers and best end pointer after it for each until we find a match.
    start_pointers = top_k_indices(start_sims, k=10)

    for pointer in start_pointers:
        pointer = max(pointer - start_shift, 0)
        end_pointer = most_similar_end_pointer(pointer, end_sims, end_shift)
        candidate_answer = prepro.text_span(context_text, spans, pointer, end_pointer)

        if candidate_answer == answer_text:
            return pointer, end_pointer

        candidate_pointers.append((pointer, end_pointer))

    # In the third case we loosen our requirement and accept answers that are roughly similar to the given answer text
    # e.g. answer_text = Japan, extracted = Japanese

    # If the end_word isn't common in the answer we take the best, otherwise we generate k answers.
    if answer_tokens.count(end_word) == 1:
        answer_sims = np.zeros(shape=(len(candidate_pointers),), dtype=np.float32)
        candidate_text = []

        for i, pointers in enumerate(candidate_pointers):
            start_pointer, end_pointer = pointers
            candidate_answer_text = prepro.text_span(context_text, spans, start_pointer, end_pointer)
            candidate_text.append(candidate_answer_text)
            answer_sims[i] = similarity_func(answer_text, candidate_answer_text)
    else:
        answer_sims = []
        candidate_pointers = []
        candidate_text = []

        for pointer in start_pointers:
            end_pointers = top_k_indices_after(pointer, end_sims, k=3)
            for end_point in end_pointers:
                candidate_pointers.append((pointer, end_point))
                candidate_answer_text = prepro.text_span(context_text, spans, pointer, end_point)
                candidate_text.append(candidate_answer_text)
                answer_sims.append(similarity_func(answer_text, candidate_answer_text))
        answer_sims = np.array(answer_sims)

    best_index = np.argmax(answer_sims)
    best_pointers = candidate_pointers[best_index]
    best_text = candidate_text[best_index]

    # If we have a subset or a superset of the answer_text.
    if answer_text in best_text or best_text in answer_text:
        return best_pointers

    return -1, -1
