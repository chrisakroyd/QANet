# QANet
## Requirements
  * Python>=3.6
  * NumPy
  * tqdm
  * TensorFlow==1.13
  * Spacy
  * Flask (only for demo)

## Usage
To download and preprocess the data, run the following commands:

```
# Download Spacy Model
python -m spacy download en_core_web_sm
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
python evaluate-v1.1.py ./data/raw/squad_v1/dev-v1.1.json ./models/logs/{run_name}/results.json
```

## Running the demo.

To run the demo, you need to have either downloaded or trained a model. To avoid polluting the repository I haven't included a pre-built version of
the demo with the project. To run the demo you need to first have node/npm installed and run the below commands.

```
npm install
npm run build-production
```

All you need to do now is run the below command to start the demo. By default the server runs on localhost:5000, not only do you
have a demo run on the port but a full API hooked up to run inference that you can use for other purposes.

```
python main.py --mode demo --run_name this_is_a_new_run_name
```

## Dataset

### SQuAD 1.1

The Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset consisting of 100,000+ question/answer pairs where each answer is a span of text from a corresponding wikipedia article. At the time of writing, a QANet ensemble is currently in 3rd place on the [SQuAD 1.1 leaderboard](https://rajpurkar.github.io/SQuAD-explorer/) but at release it achieved state-of-the-art results. Although SQuAD is a great example of an open-source public dataset, there are several cases where the given answers don't match the wikipedia articles, have several characters cut off or the given text is littered with [citation needed] tags. We deal with this via a similarity measure during preprocessing and adopt the character-character alignment approach from [BERT](https://github.com/google-research/bert) when extracting answers.

To run QANet on SQuAD, you will first need to download the dataset. The the necessary data files can be found here:

*   [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
*   [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)

Alternatively run the command ```python main.py --mode download``` and all the neccessary files will be downloaded automatically. To run on this dataset the command is:
```
python main.py --mode train --dataset squad_v1
```

### SQuAD 2.0

SQuAD 2.0 is the next iteration of the SQuAD dataset that introduces 43,498 negative examples, questions which have no answer within the given text. This expands the total size of the dataset to 130,319 and introduces an is_impossible key and possible answers that could answer the question but do not. An experimental version of this repository which works with squad-2.0 can be found on its own branch [here](https://github.com/ChristopherAkroyd/QANet/tree/squad-2.0).

To run QANet on SQuAD, you will first need to download the dataset. The the necessary data files can be found here:

*   [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
*   [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
*   [Evaluation Script v2.0](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)

Alternatively run the command ```python main.py --mode download``` and all the neccessary files will be downloaded automatically. To run on this dataset the command is:
```
python main.py --mode train --dataset squad_v2
```

### Preprocessed Data and Pretrained Model

#### Preprocessed data
* [No augmentation Squad V1.1 data](https://drive.google.com/open?id=1N_Il8LK8LdiYpJCR_JGmfgX4t1oeVoL_).

#### Pre-trained models
* 30k Training steps, 1 attention head 72.0/80.7 EM/F1 [download here]().
* 60k Training steps, 1 attention head, 72.1/80.8 EM/F1 [download here]().

Preprocessed data will be available once I've completed the implementation of the data augmentation section of the paper. A model trained for 28k steps with dev EM/F1 score of 69.4/78.8 can be downloaded from here: INSERT LINK TO CLOUD DOWNLOAD

## Results
Here are the collected results from this repository and the original paper.

|      Model     | Training Steps | Size | Attention Heads | Data Size (aug) |  EM  |  F1    | Time |
|:--------------:|:--------------:|:----:|:---------------:|:---------------:|:----: |:----:  |:----:|
|This repository |     30,000     |  128  |        1        |   87k (no aug)  | 72.0 | 80.7  | 3h 07m|
|This repository |     60,000     |  128  |        1        |   87k (no aug)  | 72.1 | 80.8  | 6h 32m |
|This repository |     150,000     |  128  |        8        |   87k (no aug)  | 73.1| 82.3 | 16h 55m|
|This repository + ELMo Fixed |     14,000     |  128  |        1        |   87k (no aug)  | 74.9 | 83.0  | 2h 15m |
|This repository + ELMo Finetuned |     13,000     |  128  |        1        |   87k (no aug)  | 75.2 | 83.3  | 5h 12m |
|[Original Paper](https://arxiv.org/pdf/1804.09541.pdf)|     -     |  128 |        8        |   87k (no aug)  |  NA  | 77.0 | 3h 00m
|[Original Paper](https://arxiv.org/pdf/1804.09541.pdf)|     150,000    |  128 |        8        |   87k (no aug)  | 73.6 | 82.7 | 18h 00m
|[Original Paper](https://arxiv.org/pdf/1804.09541.pdf)|     340,000    |  128 |        8        |    240k (aug)   | 75.1 | 83.8 | -

Results + train time for single-headed runs were measured through tensorboard on a computer running Windows 10 Education with a Nvidia GTX 1080 Ti, Ryzen Threadripper 1900X and 32GB RAM. Runs with 8 attention heads were trained on a Google Cloud n1-standard-8 instance with a V100 GPU. ELMo runs used max_tokens of 300 instead of 400 used in the original paper

The ELMo results could very easily be improved with some parameter tuning/a run longer than 15000 steps/max_tokens > 300 but I currently have not had the time/resources to do this myself.

## Implementation Notes
  * This repository makes heavy use of tf.keras API for code readability and future compatability reasons as TensorFlow is standardising upon this for TF v2.0.
  * Fast and efficient train / test / demo pipeline that allows training at speeds close to those reported in the original paper and takes up <100 MB on disk (excluding embeddings).
  * Uses character-level convolution in conjunction with pre-trained word embeddings and highway layers to embed words built on [Yoon Kim's work on character-cnns](https://arxiv.org/pdf/1508.06615.pdf) and the approach employed by [Bi-directional Attention Flow](https://arxiv.org/pdf/1611.01603.pdf)
  * Employs the use of [DCN](https://arxiv.org/pdf/1611.01604.pdf) style Query-to-Context attention and Context-to-Query attention for memory-saving computation.
  * Exponential Moving Average (EMA) is employed over all trainable variables to improve test-time accuracy.

## Differences to Original Paper
  * In the original paper, each word is padded or truncated to 16 characters in length before embedding, this implementation pads or truncates to the max character length within the batch.
  * The original paper doesn't specify how they handled using pre-trained embeddings with trainable unknown tokens (<UNK>), this implementation adopts a original approach.
  * A linear learning rate warmup rate is used versus the inverse exponential used in the paper.
  * Position encoding takes place at the start of each stack vs the start of each block as in the original paper.
  * Weights are initialized with a truncated normal initializer with a standard deviation of 0.02.
  * Currently supports both fixed and fine-tunable [ELMo embeddings](https://allennlp.org/elmo).

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
- [x] Official script evaluation.
- [x] Character-to-character answer alignment from BERT (Devlin et al. 2018)
- [ ] SQuAD 2.0 support.
- [x] ELMo support.
- [ ] Tensor processing unit (TPU) and tf.estimator Support.


#### Demo
- [x] Demo UI.
- [x] Start / End Pointer word heatmap.
- [ ] Tensorflow.js + running in the browser (long term goal).

#### Docs
- [x] Actually add documentation.

## FAQ

#### Help, I've get Errors!
First step is to check that you meet the exact [requirements](#requirements) for this implementation. Although tensorflow 1.12 is a hard requirement i have tested and managed to run the code in tensorflow versions 1.5 - 1.12 with minor API changes. If you still have errors, please submit a [Github Issue](https://github.com/ChristopherAkroyd/QANet/issues).

#### Can I use my own data?
Yes! If your data is formatted exactly the same as a SQuAD file everything will run perfectly, otherwise you would need to make modifications to ```/preprocess.py``` to accommodate for the differences.

#### Does this work with FastText vectors or other embedding schemes?
Yes! As long as the file storing the vectors is formatted like a GloVE or FastText file the only major issue will be that the tokenization of the text may not line up exactly with the word embeddings.

#### I am getting out-of-memory errors, what is wrong?

This issue occurs because gradient calculation for the multi-head attention can be quite memory intensive, especially
when used in tandem with a high max_tokens. The best way to avoid this is to, reduce the character dimension, hidden size
or the batch size or max_tokens.

If this doesn't work or you'd still like to train with full parameters, pass the command line flag `--low_memory`. This is an **_experimental_** feature that tells the neural network to recompute the forward pass of every multi-head attention layer during backpropagation in order to save GPU memory at the cost of performance (≈40% slower) but allows the 8-headed variant to be trained on a 11GB GPU. As this is experimental, there are **_no guarantees_** that an 8-headed variant trained with recomputed gradients will match the performance of one trained using standard backprop.

#### What is Spacy being used for?
Spacy provides much better tokenization out of the box vs nltk or simply splitting on whitespace.

#### Whats the webpack.common.js file for and why is there a folder full of javascript files?
I implemented the demo functionality for the repository with React + Redux with the idea that in the future I could try and get this model running in the browser with [Tensorflow.js](https://js.tensorflow.org/) . If you feel that JavaScript is the devil or React isn't your favourite framework you can safely ignore all of this and it'll sit inert on your hard drive. If you wonder why it was neccesary at all, the [Google Dev Summit 2018](https://www.youtube.com/watch?v=YB-kfeNIPCE) provides some good justifications for Machine Learning on the frontend, the React + Redux choice was motivated by reducing what I'd need to do later anyway (state management, app flow etc.) by using familiar tools.


## Contact information

For help or issues using this repo, please submit a GitHub issue.