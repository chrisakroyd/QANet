import random
from src import preprocessing as prepro


def get_examples(contexts, queries, num_examples=1000):
    """ Gets a subsample of the contexts/queries.  """
    shuffled = list(queries.values())
    random.shuffle(shuffled)
    shuffled = shuffled[:num_examples]
    examples = [{'context': contexts[data['context_id']]['text'], 'query': data['text']} for data in shuffled]
    return examples


def convert_to_indices(text, tokens):
    """ Maps each token to its character offset within the text. """
    current = 0
    spans = []

    for token in tokens:
        next_toke_start = text.find(token, current)

        # We normalize all dash characters (e.g. en dash, em dash to -) which requires special handling when mapping.
        # TODO: @cakroyd look at this and improve implementation.
        if len(token) == 1 and prepro.is_dash(token):
            if prepro.is_dash(text[current]):
                current = current
            elif prepro.is_dash(text[current + 1]):
                current = current + 1
        else:
            current = next_toke_start

        if current == -1:
            print('Token {} cannot be found'.format(token))
            raise ValueError('Could not find token.')

        spans.append((current, current + len(token)))
        current += len(token)
    return spans
