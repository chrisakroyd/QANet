# Project Overview
This is an implemenation of the paper [QANet](https://arxiv.org/pdf/1804.09541.pdf)
## Running from scratch

## Adding a new dataset

## File Structure

````
├── data
│   ├── embeddings     <- Default folder for saving embeddings.
│   ├── processed      <- Folder where we store pre-processed data.
│   └── raw            <- Original data folder.
│   └── defaults.json  <- Where we store model parameters, we save a copy of this with each
│                         checkpoint for easy save/restoring at a later date.
│
├── docs               <- Collection of hopefully useful documents.
│
├── models
│   ├── checkpoints    <- Where we store checkpoint files.
│   └── logs           <- Where we store logs and tensorboard log files.
│
├── notebooks          <- Folder for Jupyter notebooks.
│
├── src                <- Source code for use in this project.
│   ├── data_aug       <- Module for augmenting data e.g. back-translation.
│   ├── demo_ui        <- Demo UI source code (Javascript etc.)
│   ├── layers         <- Keras layer definitions
│   ├── loaders        <- Module containing scripts for loading different datasets e.g. squad v1, squadv2
│   ├── models         <- Module containing complete Keras models e.g. QANet
│   ├── preprocessing  <- Module containing preprocessing scripts.
│   ├── util           <- Utility code for loading embeddings + generating filepaths.
│   ├── config.py      <- Script for command line parameter loading with defaults from defaults.json
│   ├── constants.py   <- File containing all constant values used throughout this project.
│   ├── demo_utils.py  <- Utilities for generating response bodies.
│   ├── metrics.py     <- Code for calculating F1 + EM metrics.
│   ├── pipeline.py    <- Code for loading train/val files + feeding into the model.
│   └── train_utils.py <- Train related utilities, EMA, L2 + learning rate schedulers.
│
├── demo.py            <- Runs the Flask Demo server.
├── download.py        <- Downloads embeddings + data.
├── test.py            <- Runs test mode, outputting metrics and a results file.
├── train.py           <- Runs train mode,
├── main.py            <- Runs any mode set with the --mode flag.
├── requirements.txt
├── LICENSE
└── README.md
````

## Config Options

````
--answer_limit: Max length of a answer.
    (default: '30')
    (an integer)
  --attn_dropout: Attention dropout rate.
    (default: '0.1')
    (a number in the range [0.0, 1.0])
  --batch_size: Batch Size
    (default: '32')
    (an integer)
  --beta1: Beta 1 parameter of adam optimizer.
    (default: '0.8')
    (a number in the range [0.0, 1.0])
  --beta2: Beta 2 parameter of adam optimizer.
    (default: '0.999')
    (a number in the range [0.0, 1.0])
  --[no]bucket: Whether to use bucketing (used in paper).
    (default: 'true')
  --bucket_ranges: Ranges for bucketing (if enabled).
    (default: '')
    (a comma separated list)
  --bucket_size: Size of a bucket (If no bucket ranges given).
    (default: '40')
    (an integer)
  --char_dim: Dimensionality of the character output embeddings
    (default: '64')
    (an integer)
  --char_limit: Max number of characters in a word.
    (default: '16')
    (an integer)
  --checkpoint_every: After how many steps do we save a checkpoint.
    (default: '1000')
    (an integer)
  --data_dir: Directory to save pre-processed word/char embeddings, indexes and
    data.
    (default: './data/')
  --dataset: Which dataset to use, e.g. Squad or MS Marco.
    (default: 'squad_v1')
  --demo_server_port: Port on which to serve/receive requests.
    (default: '5000')
    (an integer)
  --dist_dir: Out path for demo code.
    (default: './dist')
  --dropout: Dropout rate.
    (default: '0.1')
    (a number in the range [0.0, 1.0])
  --ema_decay: Exponential moving average decay rate.
    (default: '0.999')
    (a number in the range [0.0, 1.0])
  --embed_dim: Dimensionality of the input embeddings
    (default: '300')
    (an integer)
  --embed_encoder_blocks: Number of blocks in the embedding_encoder.
    (default: '1')
    (an integer)
  --embed_encoder_convs: Number of conv layers in each block of the embed
    encoder.
    (default: '4')
    (an integer)
  --embed_encoder_kernel_width: Kernel width of each conv layer of the embed
    encoder.
    (default: '7')
    (an integer)
  --embeddings_path: Path to Glove/embedding file.
    (default: './data/embeddings/glove.840B.300d.txt')
  --epsilon: Value for epsilon.
    (default: '1e-07')
    (a number)
  --ff_inner_size: Number of units in the first layer of a feed forward block.
    (default: '128')
    (an integer)
  --gradient_clip: Clip by global norm value.
    (default: '5.0')
    (a number)
  --heads: Number of heads used for multi-head attention.
    (default: '1')
    (an integer)
  --[no]help: Print flag help
    (default: 'false')
  --hidden_size: Number of hidden units to use.
    (default: '128')
    (an integer)
  --l2: L2 weight decay.
    (default: '3e-07')
    (a number)
  --learn_rate: Learning rate.
    (default: '0.001')
    (a number)
  --max_chars: Max chars to be included in the word index.
    (default: '2500')
    (an integer)
  --max_prefetch: Max number of prefetched batches.
    (default: '5')
    (an integer)
  --max_tokens: Max length of the input paragraph.
    (default: '400')
    (an integer)
  --max_words: Max words to be included in the word index.
    (default: '150000')
    (an integer)
  --min_char_occur: Min times a character must be seen to be included in the
    char index.
    (default: '-1')
    (an integer)
  --min_word_occur: Min times a word must be seen to be included in the word
    index.
    (default: '-1')
    (an integer)
  --mode: Train/test/demo.
    (default: 'train')
  --model_encoder_blocks: Number of blocks in the model_encoder.
    (default: '7')
    (an integer)
  --model_encoder_convs: Number of conv layers in each block of the model
    encoder.
    (default: '2')
    (an integer)
  --model_encoder_kernel_width: Kernel width of each conv layer of the model
    encoder.
    (default: '5')
    (an integer)
  --models_dir: Directory to save the models, logs and answer files.
    (default: './models/')
  --oov_token: Which word represents out of vocab words.
    (default: '<OOV>')
  --parallel_calls: Number of parallel calls for the pipeline.
    (default: '-1')
    (an integer)
  --plateau_steps: Only used when `use_cosine_decay` is True. Number of steps to
    hold the learning rate for before decaying.
    (default: '0')
    (an integer)
  --run_name: Name for this run of training.
    (default: 'cakroyd_qanet')
  --run_val_every: After how many steps do we calculate EM/F1 scores.
    (default: '1000')
    (an integer)
  --shuffle_buffer_size: Buffer size of the dataset shuffle function.
    (default: '15000')
    (an integer)
  --tf_record_buffer_size: Buffer size of a tf_record dataset.
    (default: '1024')
    (an integer)
  --train_steps: Number of training steps to perform.
    (default: '60000')
    (an integer)
  --trainable_words: Which words should have trainable embeddings.
    (default: '<OOV>')
    (a comma separated list)
  --[no]use_cosine_decay: Whether or not to use cosine decay on the learning
    rate.
    (default: 'false')
  --[no]use_elmo: Whether to use ELMo embeddings.
    (default: 'false')
  --warmup_scheme: Learning rate warmup scheme.
    (default: 'inverse_exp')
  --warmup_steps: Number of warmup steps.
    (default: '1000')
    (an integer)
  --[no]write_answer_file: Whether or not to write an out file with predictions.
    (default: 'false')
````


## How the pipeline works
The training and validation pipeline is built with the tf.data API and lookup tables, lookup tables for a text input pipeline may be unfamiliar so I will be giving a brief explanation. During pre-processing we save the context and query as a sequence of string tokens and perform an index lookup when we first load the data.

We first create a lookup table from a vocab (in this repo we create two, one for words, one for characters):
`word_table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(vocab, dtype=tf.string), default_value=len(vocab) - 1)`

We then convert a sequence of string tokens to a sequence of word indices and add 1 so that 0 can be treated as a pad token.
`context_words = word_table.lookup(fields['context_tokens']) + 1`

We get the characters by using string split and perform the lookup in the exact same way. The only difference is that as a result of the string split the lookup returns a sparse tensor which we must convert to be dense.

````
context_chars = tf.string_split(fields['context_tokens'], delimiter='')
context_chars = char_table.lookup(context_chars), default_value=-1) + 1
context_chars = tf.sparse.to_dense(context_chars)
````

So why use this method? Instead of having to save two matrices of shape [max_tokens], [max_tokens, char_limit] we end up only needing to save one which cuts filesize by two thirds and decreases load time by the same amount. By caching the dataset we only need to perform this once on the first epoch, negating any performance impact from the lookups.
