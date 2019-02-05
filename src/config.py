import tensorflow as tf


def gpu_config():
    """ Function that creates a GPU config.
        Returns:
            A `tf.ConfigProto` instance.
    """
    config = tf.ConfigProto(
        allow_soft_placement=True,
    )

    config.gpu_options.allow_growth = True

    return config


def model_config(defaults):
    """ Function that creates all the flags for this project, defaults are filled from defaults.json .
        Args:
            defaults: An object containing the default parameters.
        Returns:
            A `tf.flags` instance.
    """
    flags = tf.flags
    flags.DEFINE_boolean('help', False, 'Print flag help')
    # Mode to run in, e.g. train, test.
    flags.DEFINE_string('mode', defaults.mode, 'Train/test/demo.')
    flags.DEFINE_string('dataset', defaults.dataset, 'Which dataset to use, e.g. Squad or MS Marco.')
    # Adds a name for this run.
    flags.DEFINE_string('run_name', defaults.run_name, 'Name for this run of training.')
    # Within these flags we define where to find the original GLoVe/FastText embeddings and where to find/save data.
    flags.DEFINE_string('embeddings_path', defaults.embeddings_path, 'Path to Glove/embedding file.')
    flags.DEFINE_string('data_dir', defaults.data_dir,
                        'Directory to save pre-processed word/char embeddings, indexes and data.')
    flags.DEFINE_string('raw_data_dir', defaults.raw_data_dir,
                        'Directory to save/load raw data from.')
    # Where we save logs, models or whether we write answers. Answer file saves in data_save_dir.
    flags.DEFINE_string('models_dir', defaults.models_dir, 'Directory to save the models, logs and answer files.')
    flags.DEFINE_boolean('write_answer_file', defaults.write_answer_file,
                         'Whether or not to write an out file with predictions.')
    # Dimension for the word + char embeddings. Char embeddings are generated during the pre-processing stage.
    flags.DEFINE_integer('embed_dim', defaults.embed_dim, 'Dimensionality of the input embeddings')
    flags.DEFINE_integer('char_dim', defaults.char_dim, 'Dimensionality of the character output embeddings')
    # Max lengths for context, query, answer, characters and mins for word+char occurrence.
    flags.DEFINE_integer('max_tokens', defaults.max_tokens, 'Max length of the input paragraph.')
    flags.DEFINE_integer('answer_limit', defaults.answer_limit, 'Max length of a answer.')
    flags.DEFINE_integer('char_limit', defaults.char_limit, 'Max number of characters in a word.')
    flags.DEFINE_integer('max_words', defaults.max_words, 'Max words to be included in the word index.')
    flags.DEFINE_integer('max_chars', defaults.max_chars, 'Max chars to be included in the word index.')
    flags.DEFINE_integer('min_word_occur', defaults.min_words,
                         'Min times a word must be seen to be included in the word index.')
    flags.DEFINE_integer('min_char_occur', defaults.min_chars,
                         'Min times a character must be seen to be included in the char index.')
    # QANet paper utilises a trainable OOV token, we also allow specification of multiple trainable word embeddings.
    flags.DEFINE_string('oov_token', defaults.oov_token, 'Which word represents out of vocab words.')
    flags.DEFINE_list('trainable_words', defaults.trainable_words, 'Which words should have trainable embeddings.')
    # Flags for the pre-processing pipeline.
    flags.DEFINE_integer('shuffle_buffer_size', defaults.shuffle_buffer_size,
                         'Buffer size of the dataset shuffle function.')
    flags.DEFINE_integer('tf_record_buffer_size', defaults.tf_record_buffer_size,
                         'Buffer size of a tf_record dataset.')
    flags.DEFINE_boolean('bucket', defaults.bucket, 'Whether to use bucketing (used in paper).')
    flags.DEFINE_list('bucket_ranges', defaults.bucket_ranges, 'Ranges for bucketing (if enabled).')
    flags.DEFINE_integer('bucket_size', defaults.bucket_size, 'Size of a bucket (If no bucket ranges given).')
    flags.DEFINE_integer('parallel_calls', defaults.parallel_calls, 'Number of parallel calls for the pipeline.')
    flags.DEFINE_integer('max_prefetch', defaults.max_prefetch, 'Max number of prefetched batches.')
    flags.DEFINE_boolean('use_elmo', defaults.use_elmo, 'Whether to use ELMo embeddings.')
    # Model hyper parameters (set to QANet paper values).
    flags.DEFINE_integer('batch_size', defaults.batch_size, 'Batch Size')
    flags.DEFINE_integer('hidden_size', defaults.hidden_size, 'Number of hidden units to use.')
    flags.DEFINE_integer('ff_inner_size', defaults.ff_inner_size,
                         'Number of units in the first layer of a feed forward block.')
    flags.DEFINE_integer('heads', defaults.heads, 'Number of heads used for multi-head attention.')
    flags.DEFINE_integer('embed_encoder_blocks', defaults.embed_encoder_blocks,
                         'Number of blocks in the embedding_encoder.')
    flags.DEFINE_integer('embed_encoder_convs', defaults.embed_encoder_convs,
                         'Number of conv layers in each block of the embed encoder.')
    flags.DEFINE_integer('embed_encoder_kernel_width', defaults.embed_encoder_kernel_width,
                         'Kernel width of each conv layer of the embed encoder.')
    flags.DEFINE_integer('model_encoder_blocks', defaults.model_encoder_blocks,
                         'Number of blocks in the model_encoder.')
    flags.DEFINE_integer('model_encoder_convs', defaults.model_encoder_convs,
                         'Number of conv layers in each block of the model encoder.')
    flags.DEFINE_integer('model_encoder_kernel_width', defaults.model_encoder_kernel_width,
                         'Kernel width of each conv layer of the model encoder.')
    # Flags for train hyper params e.g. dropout, l2, gradient ema decay values (set to QANet paper values).
    flags.DEFINE_float('dropout', defaults.dropout, 'Dropout rate.', lower_bound=0.0, upper_bound=1.0)
    flags.DEFINE_float('attn_dropout', defaults.attn_dropout, 'Attention dropout rate.',
                       lower_bound=0.0, upper_bound=1.0)
    flags.DEFINE_float('l2', defaults.l2, 'L2 weight decay.', lower_bound=0.0)
    flags.DEFINE_float('gradient_clip', defaults.gradient_clip, 'Clip by global norm value.', lower_bound=0.0)
    flags.DEFINE_float('learn_rate', defaults.learn_rate, 'Learning rate.', lower_bound=0.0)
    flags.DEFINE_float('beta1', defaults.beta1, 'Beta 1 parameter of adam optimizer.', lower_bound=0.0, upper_bound=1.0)
    flags.DEFINE_float('beta2', defaults.beta2, 'Beta 2 parameter of adam optimizer.', lower_bound=0.0, upper_bound=1.0)
    flags.DEFINE_float('epsilon', defaults.epsilon, 'Value for epsilon.')
    flags.DEFINE_float('ema_decay', defaults.ema_decay, 'Exponential moving average decay rate.',
                       lower_bound=0.0, upper_bound=1.0)
    # Train specific flags e.g. number of steps, early stop, eval period.
    flags.DEFINE_integer('train_steps', defaults.train_steps, 'Number of training steps to perform.')
    flags.DEFINE_integer('warmup_steps', defaults.warmup_steps, 'Number of warmup steps.')
    flags.DEFINE_integer('plateau_steps', defaults.plateau_steps,
                         'Only used when `use_cosine_decay` is True. Number of steps to hold the learning rate for before decaying.')
    flags.DEFINE_boolean('use_cosine_decay', defaults.use_cosine_decay,
                         'Whether or not to use cosine decay on the learning rate.')
    flags.DEFINE_string('warmup_scheme', defaults.warmup_scheme, 'Learning rate warmup scheme.')
    flags.DEFINE_integer('checkpoint_every', defaults.checkpoint_every, 'After how many steps do we save a checkpoint and calculate EM/F1 scores.')
    return flags
