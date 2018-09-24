import os
import random
import re
from tqdm import tqdm
from ftfy import fix_text
from unidecode import unidecode
from util.Tokenizer import Tokenizer
from util.embeddings import load_embeddings, generate_matrix
from util.util import save_json, load_json, save_embeddings, index_from_list

oov_token = '<oov>'
max_words = 150000
trainable_words = [oov_token]

# Glove path
glove_path = 'E:/data_sets/glove/glove.840B.300d.txt'
# Paths to save Datasets
train_elmo_path = '../data/squad/train_elmo.json'
train_indexed_path = '../data/squad/train_indexed.json'
train_contexts_path = '../data/squad/train_contexts.json'
train_answers_path = '../data/squad/train_answers.json'
dev_elmo_path = '../data/squad/val_elmo.json'
dev_indexed_path = '../data/squad/val_indexed.json'
dev_contexts_path = '../data/squad/val_contexts.json'
dev_answers_path = '../data/squad/val_answers.json'
# Paths to save embeddings/indexes
word_index_save_path = './QANet/data/word_index.json'
char_index_save_path = './QANet/data/char_index.json'
trainable_word_index_save_path = './QANet/data/trainable_word_index.json'
word_embeddings_save_path = './QANet/data/embeddings.txt'
char_embeddings_save_path = './QANet/data/char_embeddings.txt'
trainable_embeddings_save_path = './QANet/data/trainable_embeddings.txt'
# Dims
word_embed_dims = 300
# char_embed_dims = 200
char_embed_dims = 64
# Other params
CONTEXT_LIMIT = 400
QUESTION_LIMIT = 50
# Regexes
apostrophe = re.compile(r"('')")
apostrophe_like = re.compile(r"(``)")


def convert_idx(text, tokens):
    current = 0
    spans = []
    text = unidecode(fix_text(text))
    text = text.replace('>', '')
    text = text.replace('*', '')

    for token in tokens:
        if token == '``':
            token = '"'
        current = text.find(token, current)
        if current < 0:
            print(text)
            print(tokens)
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def fit_on_squad(data_set, tokenizer):
    for data in data_set['data']:
        for question_answer in data['paragraphs']:
            context = question_answer['context']
            context = apostrophe.sub('" ', context)
            context = apostrophe_like.sub('" ', context)

            tokenizer.fit_on_texts(context)

            for qa in question_answer['qas']:
                ques = qa['question'].replace("''", '" ').replace("``", '" ')
                tokenizer.fit_on_texts(ques)

    return tokenizer


def preprocess(data_set, tokenizer, context_limit, question_limit):
    elmo = []
    indexed = []
    answers = {}
    contexts = {}
    context_id = 1
    answer_id = 1

    for data in tqdm(data_set['data']):
        for question_answer in data['paragraphs']:
            context = unidecode(fix_text(question_answer['context']))
            context = apostrophe.sub('" ', context)
            context = apostrophe_like.sub('" ', context)

            context_elmo = tokenizer.tokenize(context)
            context_words, context_chars, context_length = tokenizer.texts_to_sequences(context,
                                                                                        max_words=context_limit,
                                                                                        numpy=False, pad=False)
            # Tokenizer wraps in a list (len==1) so we need the last entry
            context_words = context_words[-1]
            context_chars = context_chars[-1]

            # Skip if outside of context limit
            if context_length[-1] > context_limit:
                continue

            spans = convert_idx(context, tokenizer.tokenize(context))

            for qa in question_answer['qas']:
                question = qa['question'].replace("''", '" ').replace("``", '" ')
                question_elmo = tokenizer.tokenize(question)
                question_words, question_chars, question_length = tokenizer.texts_to_sequences(question,
                                                                                               max_words=question_limit,
                                                                                               numpy=False, pad=False)
                # Tokenizer wraps in a list (len==1) so we need the last entry
                question_words = question_words[-1]
                question_chars = question_chars[-1]

                # Skip if its outside of the limits.
                if question_length[-1] > question_limit:
                    continue

                answer_starts, answer_ends = [], []
                answer_texts = []

                for answer in qa['answers']:
                    answer_text = answer['text']
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer_text)
                    answer_texts.append(answer_text)
                    answer_span = []

                    for i, span in enumerate(spans):
                        if not (answer_end <= span[0] or answer_start >= span[1]):
                            answer_span.append(i)

                    if len(answer_span) < 1:
                        continue

                    answer_starts.append(answer_span[0])
                    answer_ends.append(answer_span[-1])

                if len(answer_starts) < 1 and len(answer_ends) < 1:
                    print('Skipping {}...'.format(qa['id']))
                    continue

                indexed.append({
                    'context_words': context_words,
                    'context_chars': context_chars,
                    'question_words': question_words,
                    'question_chars': question_chars,
                    'answer_starts': answer_starts[-1],
                    'answer_ends': answer_ends[-1],
                    'answer_id': answer_id,
                })
                # For elmo we just need the tokens as opposed to word lookup indexes
                elmo.append({
                    'context_id': context_id,
                    'context_words': context_elmo,
                    'question_words': question_elmo,
                    'answer_starts': answer_starts[-1],
                    'answer_ends': answer_ends[-1],
                    'answer_id': answer_id,
                })

                answers[answer_id] = {
                    'id': qa['id'],
                    'answer_id': answer_id,
                    'context_id': context_id,
                    'answers': answer_texts,
                }

                answer_id += 1

            contexts[context_id] = {
                'id': context_id,
                'context': context,
                'word_spans': spans,
            }

            context_id += 1

    random.shuffle(indexed)
    print("{} questions in total".format(len(indexed)))
    return indexed, elmo, contexts, answers


def process(data_dir, vocab_path=None, trainable_words=['<oov>'], context_limit=CONTEXT_LIMIT, question_limit=QUESTION_LIMIT):
    train_path = os.path.join(data_dir, 'train-v1.1.json')
    dev_path = os.path.join(data_dir, 'dev-v1.1.json')

    vocab = load_json(vocab_path) if vocab_path is not None else None

    tokenizer = Tokenizer(max_words=max_words + 1,
                          vocab=vocab,
                          lower=False,
                          oov_token=oov_token,
                          min_occurence=-1,
                          trainable_words=trainable_words,
                          filter_punct=set(list(' #*/@[\]^_`{|}~')))

    train = load_json(train_path)
    dev = load_json(dev_path)

    tokenizer = fit_on_squad(train, tokenizer)
    tokenizer = fit_on_squad(dev, tokenizer)

    train_indexed, train_elmo, train_contexts, train_answers = preprocess(train, tokenizer,
                                                                          context_limit=context_limit,
                                                                          question_limit=question_limit)
    dev_indexed, dev_elmo, dev_contexts, dev_answers = preprocess(dev, tokenizer,
                                                                  context_limit=context_limit,
                                                                  question_limit=question_limit)

    word_index = tokenizer.word_index
    char_index = tokenizer.char_index
    trainable_word_index = index_from_list(trainable_words)

    embedding_matrix, trainable_matrix = load_embeddings(path=glove_path,
                                                         word_index=word_index,
                                                         embedding_dimensions=word_embed_dims,
                                                         trainable_embeddings=trainable_words)

    char_matrix = generate_matrix(index=char_index, embedding_dimensions=char_embed_dims)

    # Save the generated data
    save_json(train_indexed_path, train_indexed, format_json=False)
    save_json(train_elmo_path, train_elmo, format_json=False)
    save_json(train_contexts_path, train_contexts, format_json=False)
    save_json(train_answers_path, train_answers)
    save_json(dev_indexed_path, dev_indexed)
    save_json(dev_elmo_path, dev_elmo)
    save_json(dev_contexts_path, dev_contexts, format_json=False)
    save_json(dev_answers_path, dev_answers)
    # Save the word index mapping of word:index for both the pre-trained and trainable embeddings.
    save_json(word_index_save_path, word_index)
    save_json(char_index_save_path, char_index)
    save_json(trainable_word_index_save_path, trainable_word_index)
    # Save the trainable embeddings matrix.
    save_embeddings(trainable_embeddings_save_path, trainable_matrix, trainable_word_index)
    # Save the full embeddings matrix
    save_embeddings(word_embeddings_save_path, embedding_matrix, word_index)
    save_embeddings(char_embeddings_save_path, char_matrix, char_index)


if __name__ == "__main__":
    process(data_dir='E:/data_sets/squad', vocab_path='../data/embeddings/vocab.json')
