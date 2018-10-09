# Description
![](https://img.shields.io/badge/Python-2.7-brightgreen.svg)

This project is a core library for
[RuSentRel](https://github.com/nicolay-r/RuSentRel) dataset processing.
This library provides API for synonyms, news, opinions, entities files reading.

## Source

### Embeddings

Represents a wrapper over [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) model api of [gensim](https://radimrehurek.com/gensim/) library.
This core provides an additional wrappers for:
* News collection from [rusvectores](http://rusvectores.org/ru/models/), which has specific pos prefixes for words of vocabulary;
* Wrapper for additional punctuation signs (tokens) in text, i.e. `":", ";", ".", "!"` etc.

## Installation

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
