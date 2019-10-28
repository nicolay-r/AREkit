# Description
![](https://img.shields.io/badge/Python-2.7-brightgreen.svg)
![](https://img.shields.io/badge/Tensorflow-1.14.0-yellowgreen.svg)

**arekit** -- is a python toolkit (library) for sentiment **attitudes relation extraction** (ARE) task.
Large news and analytical articles shares a large amount of opinions conveyed as by author towards 
mentioned entities/events, and also between mentioned entities, i.e. from **subject** towards **object**.
The contribution of this library is as follows: 

* Brings the task of sentiment attitudes extraction at 
fundamental level by providing a common types/structures that are important for domain research;
Please refer to `common/` source directory for more details.

* Provides a necessary toolset to perform natural language processing (NLP): 
text parsers, 
syntax processing, 
named entity recognition (NER), 
part of speech tagging (POS),
stemmer;
Please refer to `processing/` source directory for more details.

* Provides implemented ML models `networks/` that are intended for automatic sentiment relation extraction (RE) 
on document level.

Structuraly, the fundamental task representation could be departed into following domains: 

1. **Named Entity Recognition** -- to extract mentioned named entities;
3. **Coreference Search**, or Entity Linking -- to match synonymous entities 
(considering SynonymsCollection at `common/`);
2. **Relation Extraction** -- to extract Subject->Object sentiment relation type
[[domain review](https://github.com/roomylee/awesome-relation-extraction)];

## Neural Networks

### Convolutional Neural Networks

#### CNN

#### Piecewise CNN

### Recurrent Neural Networks (Sequence-Based Text Presentation)

### Attention Architectures

#### IAN

Includes:
* Frame aspect based implementation [[code]](networks/context/architectures/ian_frames.py);
* Attitude ends aspect based implementation;
> NOTE: Experiments with RuSentRel results in an application of base Optimizer instead of 
`tf.train.AdamOptimizer(learning_rate=learning_rate)` oprimizer. The latter stucks training process.

#### Att-BiLSTM

### Training Approaches
    
1. Single Sentence Training

2. Multiple Sentence Training

#### Layers Regularization

We utilize 'L2'-regularization for layers and then combine with the ordinary loss 
([stack-overflow-post](https://stackoverflow.com/questions/37107223/how-to-add-regularizations-in-tensorflow#37143333)):
```
tf.get_variable('a', regularizer=tf.contrib.layers.l2_regularizer(0.001))
loss = ordinary_loss + tf.losses.get_regularization_loss()
```

## Source

### Datasets

Provides reader for [RuSentRel](https://github.com/nicolay-r/RuSentRel) dataset.

### Embeddings

Represents a wrapper over [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) model api of [gensim](https://radimrehurek.com/gensim/) library.
This core provides an additional wrappers for:
* News collection from [rusvectores](http://rusvectores.org/ru/models/), which has specific pos prefixes for words of vocabulary;
* Wrapper for additional punctuation signs (tokens) in text, i.e. `":", ";", ".", "!"` etc.

## Processing

### Lemmatization

Available wrappers:
1. Yandex Mystem;

### Part-Of-Speech Tagger (POS)

Available wrappers:
1. Yandex Mystem;

### Named entities recognition (NER)

Provides wrappers for: 
1. [DeepPavlov](https://github.com/deepmipt/ner)
2. [Texterra](https://texterra.ispras.ru/)

### Syntax parser

Provides wrappers for:
1. INemo SyntaxNet
2. Texterra syntax parser

## Installation

> NOTE. Provide updated description for 3.3+ python.

Using [virtualenv](https://www.pythoncentral.io/how-to-install-virtualenv-python/).
Create virtual environment, suppose `my_env`, and activate it as follows:
```
virtualenv my_env
source my_env/bin/activate
```

Then install dependencies as follows:
```
pip install -r dependencies.txt
```
