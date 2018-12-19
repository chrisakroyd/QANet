# QANet
## What makes this implementation different?
## Requirements
  * Python>=3.6
  * NumPy
  * tqdm
  * TensorFlow==1.12
  * Spacy
  * Flask (only for demo)

## Dataset

### SQuAD 1.1

The Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset consisting of 100,000+ question/answer pairs where each answer is a span of text from a corresponding wikipedia article. At the time of writing, a QANet ensemble is currently in 3rd place on the [SQuAD 1.1 leaderboard](https://rajpurkar.github.io/SQuAD-explorer/) but at release it achieved state-of-the-art results. Although SQuAD is a great example of an open-source public dataset, there are several cases where the given answers don't match the wikipedia articles, have several characters cut off or the given text is littered with [citation needed] tags. We deal with this via a similarity measure during preprocessing and adopt the character-character alignment approach from [BERT](https://github.com/google-research/bert) when extracting answers.

To run QANet on SQuAD, you will first need to download the dataset. The the necessary data files can be found here:

*   [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
*   [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

Alternatively run the command ```python main.py --mode download``` and all the neccessary files will be downloaded automatically.

## Usage
To download and preprocess the data, run the following commands:

```
# Download SQuAD and Glove
python main.py --mode download
# Preprocess the data
python main.py --mode preprocess
```

To start training simply run the following command:

```
python main.py --mode train
```

Want to start a new run? just run the command:

```
python main.py --mode train --run_name this_is_a_new_run_name
```

Want to change several model parameters without wanting to type them out? Simply modify the defaults.json file in the ```/data/``` folder. A copy of this file is saved with the model checkpoints so you can keep track of run parameters and reduce errors in restoring models.

A full list of config options can be found in ```/docs/```

The default directory for tensorboard log files is `models/logs`, logs for each run will be saved in a sub-directory for each training run.

```
tensorboard --logdir=./models/logs
```

To evaluate the model with the official code, run
```
python evaluate-v1.1.py ~/data/squad/dev-v1.1.json models/logs/{run_name}/answers.json
```



### Preprocessed Data and Pretrained Model

Preprocessed data will be available once I've completed the implementation of the data augmentation section of the paper. A model trained for 28k steps with dev EM/F1 score of 69.4/78.8 can be downloaded from here: INSERT LINK TO CLOUD DOWNLOAD

## Results
Here are the collected results from this repository and the original paper.

|      Model     | Training Steps | Size | Attention Heads | Data Size (aug) |  EM  |  F1    | Time |
|:--------------:|:--------------:|:----:|:---------------:|:---------------:|:----: |:----:  |:----:|
|This repository* |     30,000     |  128  |        1        |   87k (no aug)  | 69.9 | 79.0  | 3h 9m|
|This repository* |     60,000     |  128  |        1        |   87k (no aug)  | 70.0 | 79.2  | 6h 18m |
|This repository* |     150,000     |  128  |        8        |   87k (no aug)  | 71.1| 80.3 | 17h 34m|
|[NLPLearn - QANet](https://github.com/NLPLearn/QANet) (reported)|60,000|128|1|87k (no aug)| 70.7 | 79.8 | - |
|[NLPLearn - QANet](https://github.com/NLPLearn/QANet) (measured)*|60,000|128|1|87k (no aug)| |  |
|[Original Paper](https://arxiv.org/pdf/1804.09541.pdf)|     35,000     |  128 |        8        |   87k (no aug)  |  NA  | 77.0 | 3h 00m
|[Original Paper](https://arxiv.org/pdf/1804.09541.pdf)|     150,000    |  128 |        8        |   87k (no aug)  | 73.6 | 82.7 | 18h 00m
|[Original Paper](https://arxiv.org/pdf/1804.09541.pdf)|     340,000    |  128 |        8        |    240k (aug)   | 75.1 | 83.8 | -

\* Results + train time measured through tensorboard on a computer running Windows 10 Education with 2 Nvidia GTX 1080 Ti, Ryzen Threadripper 1900X and 32GB RAM. Results gained without the use of exponential moving average variables, test-time performance is therefore likely to be higher.

## Implementation Notes
  * This repository makes heavy use of tf.keras API for code readability and future compatability reasons as TensorFlow is standardising upon this for TF v2.0.
  * Fast and efficient train / test / demo pipeline that allows training at speeds close to those reported in the original paper and takes up <100 MB on disk (excluding embeddings).
  * Uses character-level convolution in conjunction with pre-trained word embeddings and highway layers to embed words built on [Yoon Kim's work on character-cnns](https://arxiv.org/pdf/1508.06615.pdf) and the approach employed by [Bi-directional Attention Flow](https://arxiv.org/pdf/1611.01603.pdf)
  * Employs the use of [DCN](https://arxiv.org/pdf/1611.01604.pdf) style Query-to-Context attention and Context-to-Query attention for memory-saving computation.
  * Exponential Moving Average (EMA) is employed over all trainable variables to improve test-time accuracy.

## Differences to Original Paper
  * In the original paper, each word is padded or truncated to 16 characters in length before embedding, this implementation pads or truncates to the max character length within the batch.
  * The original paper doesn't specify how they handled using pre-trained embeddings with trainable unknown tokens (<UNK>), this implementation adopts a original approach which I will be detailing in a blog post at a later date.
  * A linear learning rate warmup rate is used versus the inverse exponential used in the paper.

## TODO's
#### Preprocessing
- [x] Data cleaning and unicode normalisation.
- [x] Answer extraction heuristics.
- [ ] Data augmentation by paraphrasing.
- [ ] Data augmentation by back-translation.
- [ ] Data augmentation with type swap method (Raiman & Miller, 2017).
- [ ] SQuAD 2.0 support.

#### Pipeline
- [x] TFRecord + tf.data pipeline.
- [x] Bucketing.
- [ ] SQuAD 2.0 support.

#### Train + Test
- [x] Trainable unknown (<UNK>) word vectors.
- [x] Training and testing the model.
- [x] Full model implementation following the original paper.
- [x] Train with full hyperparameters (Augmented data, 8 heads, hidden units = 128).
- [ ] Official script evaluation.
- [ ] Character-to-character answer alignment from BERT (Devlin et al. 2018)
- [ ] SQuAD 2.0 support.
- [ ] ELMo support.
- [ ] Tensor processing unit (TPU) and tf.estimator Support.


#### Demo
- [x] Demo UI.
- [x] Start / End Pointer word heatmap.
- [ ] Context-to-query and Query-to-context attention visualisation.
- [ ] Tensorflow.js + running in the browser (long term goal).

#### Docs
- [ ] Actually add documentation.

## FAQ

#### Help, I've get Errors!
First step is to check that you meet the exact [requirements](#requirements) for this implementation. Although tensorflow 1.12 is a hard requirement i have tested and managed to run the code in tensorflow versions 1.5 - 1.12 with minor API changes. If you still have errors, please submit a [Github Issue](https://github.com/ChristopherAkroyd/QANet/issues).

#### Can I use my own data?
Yes! If your data is formatted exactly the same as a SQuAD file everything will run perfectly, otherwise you would need to make modifications to ```/preprocess.py``` to accommodate for the differences.

#### Does this work with FastText vectors or other embedding schemes?
Yes! As long as the file storing the vectors is formatted like a GloVE or FastText file the only major issue will be that the tokenization of the text may not line up exactly with the word embeddings.

#### I am getting out-of-memory errors, what is wrong?

Generally this means you the model is too big to fit into GPU memory, try reducing the character dimension, hidden size or the batch size. Otherwise I'm working on adding support for [Gradient Checkpointing](https://github.com/openai/gradient-checkpointing) and a more memory-efficient multi-head attention implementation.

#### What is Spacy being used for?
Spacy provides much better tokenization out of the box vs nltk or simply splitting on whitespace.

#### Whats the webpack.common.js file for and why is there a folder full of javascript files?
I implemented the demo functionality for the repository with React + Redux with the idea that in the future I could try and get this model running in the browser with [Tensorflow.js](https://js.tensorflow.org/) . If you feel that JavaScript is the devil or React isn't your favourite framework you can safely ignore all of this and it'll sit inert on your hard drive. If you wonder why it was neccesary at all, the [Google Dev Summit 2018](https://www.youtube.com/watch?v=YB-kfeNIPCE) provides some good justifications for Machine Learning on the frontend, the React + Redux choice was motivated by reducing what I'd need to do later anyway (state management, app flow etc.) by using familiar tools.


## Contact information

For help or issues using this repo, please submit a GitHub issue.
