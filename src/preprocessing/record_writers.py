import tensorflow as tf
import random
import tensorflow_hub as hub
import numpy as np
from tqdm import tqdm
from operator import itemgetter
from src import constants, config, util


class RecordWriter(object):
    def __init__(self, max_tokens):
        """ RecordWriter class implements base functionality and utility methods for writing .tfrecord files
            Args:
                max_tokens: Maximum number of tokens per row, rows over this will be skipped by default.
        """
        self.max_tokens = max_tokens

    def float_list(self, values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def int_list(self, values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def byte_list(self, values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    def create_feature_dict(self, context, query):
        """ Extracts features and encodes them as tf.train Feature's. """
        encoded_context = [m.encode('utf-8') for m in context['tokens']]
        encoded_query = [m.encode('utf-8') for m in query['tokens']]

        features = {
            'context_tokens': self.byte_list(encoded_context),
            'context_length': self.int_list([context['length']]),
            'query_tokens': self.byte_list(encoded_query),
            'query_length': self.int_list([query['length']]),
            'answer_starts': self.int_list([query['answer_starts']]),
            'answer_ends': self.int_list([query['answer_ends']]),
            'answer_id': self.int_list([query['answer_id']]),
        }

        if 'is_impossible' in query:
            features['is_impossible'] = self.int_list([query['is_impossible']])

        return features

    def create_record(self, context, query):
        """ Creates a formatted tf.train Example for writing in a .tfrecord file. """
        features = self.create_feature_dict(context, query)
        record = tf.train.Example(features=tf.train.Features(feature=features))
        return record

    def shuffle(self, data):
        """ Takes a dict input and returns a shuffled list of its values. """
        shuffled = list(data.values())
        random.shuffle(shuffled)
        return shuffled

    def write(self, path, contexts, queries, skip_too_long=True):
        """ Writes out context + queries for a dataset as a .tfrecord file, optionally skipping rows with too many
            tokens.

            Args:
                path: Filepath to write out a .tfrecord file
                contexts: Pre-processed contexts.
                queries: Pre-processed queries.
                skip_too_long: Boolean flag for whether rows > max_tokens are skipped or included.
        """
        shuffled_queries = self.shuffle(queries)

        with tf.python_io.TFRecordWriter(path) as writer:
            for query in shuffled_queries:
                context = contexts[query['context_id']]
                num_context_tokens = context['length']
                num_query_tokens = query['length']

                if (num_context_tokens > self.max_tokens or num_query_tokens > self.max_tokens) and skip_too_long:
                    continue

                record = self.create_record(context, query)
                writer.write(record.SerializeToString())


class ContextualEmbeddingWriter(RecordWriter):
    """ Extends RecordWriter to write out contextual embeddings a long with the row of data. Contextual models,
        e.g. BERT and ELMo are large models trained in an unsupervised manner over a large amount of data, thus
        their inclusion results in a large boost in performance. This method results in fixed embedding vectors
        and increased end performance can be achieved with finetuning at the cost of memory usage + train time.

        Args:
            max_tokens: Maximum number of tokens per row, rows over this will be skipped by default.
            contextual_model: Which contextual model to use, e.g. BERT, ELMo
    """
    def __init__(self, max_tokens, contextual_model):
        super(ContextualEmbeddingWriter, self).__init__(max_tokens)
        self.session = tf.Session(config=config.gpu_config())

        if contextual_model == 'elmo':
            self.contextual_model = hub.Module(constants.Urls.ELMO, trainable=False)
        else:
            raise NotImplementedError('Only currently support ELMo as a contextual embedding module.')

        self.session.run(tf.global_variables_initializer())

        self.tokens_input = tf.placeholder(shape=(None, None,), dtype=tf.string)
        self.lengths_input = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.embed_out = self.contextual_model(inputs={'tokens': self.tokens_input, 'sequence_len': self.lengths_input},
                                               signature='tokens', as_dict=True)['elmo']

    def possibly_lower_batch_size(self, offset, lengths, base_batch_size=32):
        """ We have contexts up to 800 tokens long, this leads to OOM when we use too large of a static
            batch size, as a precaution we drop it for lengths > 350

            TODO: Objectively fragile, only tested on a 1080ti + evaluated empirically, needs to be re-done.

            Args:
                offset: Start index,
                lengths: 1-d array of lengths
                base_batch_size: Base batch size.
            Returns:
                Batch size value for a given context length.
        """
        batch_end_index = min(offset + base_batch_size, len(lengths))
        highest_length = lengths[batch_end_index - 1]  # We pre-sort by length and don't shuffle so this will be highest
        batch_size = base_batch_size

        if highest_length > 350:
            batch_size = 16

        return batch_size

    def extract_lists(self, rows, max_tokens=-1, skip_too_long=False, id_key='id'):
        """ Our data is in dicts but for convenience while batching we flatten into 3 lists to simplify . """
        all_tokens, all_lengths, all_ids, total = [], [], [], 0

        for row in rows:
            if not (row['length'] > max_tokens and skip_too_long):  # Skip rows > max_tokens
                all_tokens.append(row['tokens'])
                all_lengths.append(row['length'])
                all_ids.append(row[id_key])  # This should be universal and not specific to the queries. TODO: FIX.
                total += 1

        assert len(all_tokens) == len(all_lengths) == len(all_ids)

        return all_tokens, all_lengths, all_ids, total

    def get_batch(self, tokens, lengths, ids, offset, total, batch_size=32):
        """ Gets a batch of size n starting at offset.  """
        batch_end_index = min(offset + batch_size, total)
        batch_tokens = tokens[offset:batch_end_index]
        batch_lengths = lengths[offset:batch_end_index]
        batch_ids = ids[offset:batch_end_index]
        assert len(batch_tokens) == len(batch_lengths) == len(batch_ids)
        batch_tokens = util.pad_to_max_length(batch_tokens, batch_lengths)
        return batch_tokens, batch_lengths, batch_ids

    def create_feature_dict(self, context, query):
        """ Extends default behaviour of create_feature_dict to include embedding features. """
        features = super(ContextualEmbeddingWriter, self).create_feature_dict(context, query)
        features.update({
            'context_embedded': self.float_list(context['embedded'].reshape(-1)),
            'query_embedded': self.float_list(query['embedded'].reshape(-1))
        })
        return features

    def embed_batch(self, batch_tokens, batch_lengths):
        """ Embeds a batch with the contextual model """
        return self.session.run(self.embed_out, feed_dict={
                        self.tokens_input: batch_tokens,
                        self.lengths_input: batch_lengths,
                    })

    def cache_contexts(self, contexts, batch_size, skip_too_long):
        print('Embedding contexts...')
        rows = sorted(contexts.values(), key=itemgetter('length'))
        tokens, lengths, ids, total = self.extract_lists(rows, self.max_tokens, skip_too_long, id_key='id')
        context_cache = {}

        with tqdm(total=total) as pbar:
            # While loop used here rather than a for so we can alter batch size for the contexts (Avoids OOM)
            i = 0
            while i < total:
                batch_size = self.possibly_lower_batch_size(i, lengths, batch_size)
                batch_tokens, batch_lengths, batch_ids = self.get_batch(tokens, lengths, ids, i, total, batch_size)
                elmo_out = self.embed_batch(batch_tokens, batch_lengths)

                for row_id, embedding, length in zip(batch_ids, elmo_out, batch_lengths):
                    new_context = contexts[row_id].copy()
                    new_context['embedded'] = np.array(embedding[:length], dtype=np.float32)
                    context_cache[int(row_id)] = new_context

                i += len(batch_tokens)
                pbar.update(batch_size)

        return context_cache

    def write(self, path, contexts, queries, skip_too_long=True):
        """ Overrides default write behaviour to generate and include fixed contextual
            embeddings for each row in the dataset.

            Args:
                path: Filepath to write out a .tfrecord file
                contexts: Pre-processed contexts.
                queries: Pre-processed queries.
                skip_too_long: Boolean flag for whether rows > max_tokens are skipped or included.
        """
        context_batch_size = 32
        query_batch_size = 64
        shuffled_queries = self.shuffle(queries)

        with tf.python_io.TFRecordWriter(path) as writer:
            # ElMo is expensive, we cut embed time by ensuring short sequences are batched together via sort by length
            context_cache = self.cache_contexts(contexts, context_batch_size, skip_too_long)
            tokens, lengths, ids, total = self.extract_lists(shuffled_queries, self.max_tokens, skip_too_long,
                                                             id_key='answer_id')

            print('embedding queries...')
            with tqdm(total=total) as pbar:
                for i in range(0, total, query_batch_size):
                    batch_tokens, batch_lengths, batch_ids = self.get_batch(tokens, lengths, ids, i, total,
                                                                            query_batch_size)
                    elmo_out = self.embed_batch(batch_tokens, batch_lengths)

                    for row_id, embedding, length in zip(batch_ids, elmo_out, batch_lengths):
                        query = queries[row_id]
                        context_id = int(query['context_id'])

                        if context_id in context_cache:
                            context = context_cache[context_id]
                            # We copy the dict as we don't want extremely large elmo arrays hanging around in memory.
                            query_copy = query.copy()
                            query_copy['embedded'] = np.array(embedding[:length], dtype=np.float32)
                            record = self.create_record(context, query_copy)
                            writer.write(record.SerializeToString())

                    pbar.update(query_batch_size)

    def close(self):
        self.session.close()
