import numpy as np
import os
from math import ceil
from src.util import load_json, pad_array


def pad(words, characters, word_limit, char_limit):
    padded_words = np.zeros((word_limit, ), dtype=np.int32)
    padded_chars = np.zeros((word_limit, char_limit, ), dtype=np.int32)

    padded_words[:len(words)] = words
    padded_chars[:len(words)] = [pad_array(char_arr, char_limit) for char_arr in characters]

    return padded_words, padded_chars


def load_squad_set(data, context_limit, question_limit, char_limit):
    data_size = len(data)
    context_words = np.zeros((data_size, context_limit), dtype=np.int32)
    context_chars = np.zeros((data_size, context_limit, char_limit), dtype=np.int32)
    question_words = np.zeros((data_size, question_limit), dtype=np.int32)
    question_chars = np.zeros((data_size, question_limit, char_limit), dtype=np.int32)
    answer_starts = np.zeros((data_size, ), dtype=np.int32)
    answer_ends = np.zeros((data_size, ), dtype=np.int32)
    answer_ids = np.zeros((data_size, ), dtype=np.int32)

    for i, row in enumerate(data):
        row_context_words = row['context_words']
        row_context_chars = row['context_chars']
        row_question_words = row['question_words']
        row_question_chars = row['question_chars']

        context_words[i], context_chars[i] = pad(row_context_words, row_context_chars, context_limit, char_limit)
        question_words[i], question_chars[i] = pad(row_question_words, row_question_chars, question_limit, char_limit)

        answer_starts[i] = row['answer_starts']
        answer_ends[i] = row['answer_ends']

        answer_ids[i] = row['answer_id']

    return context_words, context_chars, question_words, question_chars, answer_starts, answer_ends, answer_ids


def load_context_answers(path):
    train_context_path = os.path.join(path, 'train_contexts.json')
    train_answer_path = os.path.join(path, 'train_answers.json')
    val_context_path = os.path.join(path, 'val_contexts.json')
    val_answer_path = os.path.join(path, 'val_answers.json')

    train_context = load_json(train_context_path)
    train_answers = load_json(train_answer_path)
    val_context = load_json(val_context_path)
    val_answers = load_json(val_answer_path)
    return train_context, train_answers, val_context, val_answers


def load_squad(train_path,
               val_path,
               question_limit,
               context_limit,
               char_limit,
               max_examples=None):
    # Read in the file, limit to max_examples and then eval the conversations
    train = load_json(train_path)
    dev = load_json(val_path)

    if max_examples is not None:
        train = train[:ceil(len(train) * (max_examples / len(train)))]

    train_set = load_squad_set(train, context_limit, question_limit, char_limit)
    val_set = load_squad_set(dev, context_limit, question_limit, char_limit)

    return train_set, val_set
