# AREkit 0.20.5

<p align="center">
    <img src="logo.png"/>
</p>

**AREkit** -- is a python toolkit for **sentiment attitude extraction** task.

## Description

Large news and analytical articles shares a large amount of opinions conveyed as by author towards
mentioned entities/events, and also between mentioned entities, i.e. from **subject** towards **object**.

* [Contribution](contrib) directory --  is a contribution in Sentiment Attitude Extraction domain;

* [Common](common) directory -- brings the sentiment attitudes extraction task at a
fundamental level by providing common types/structures that are important for domain research;

* [Processing](processing) directory -- provides a necessary toolset to perform natural language processing (NLP):
text parsers,
syntax processing,
named entity recognition (NER),
part of speech tagging (POS),
stemmer;

* [Networks](contrib/networks/core) directory -- Provides both neural network model implementation (in Tensorflow) 
intended for the automatic sentiment relation extraction (RE)
on document level.

* [Bert](contrib/bert/core) directory -- Tsv encoders for the related task which is assumes to be applied for experiments with 
BERT language model. 

Structurally, the fundamental task representation could be departed into following domains:

1. **Named Entity Recognition** -- to extract mentioned named entities;
3. **Coreference Search**, or Entity Linking -- to match synonymous entities
(considering SynonymsCollection at `common/`);
2. **Relation Extraction** -- to extract Subject->Object sentiment relation type
[[domain review](https://github.com/roomylee/awesome-relation-extraction)];

## Dependencies

List of the toolset dependencies is as follows:

* ![](https://img.shields.io/badge/Python-2.7-brightgreen.svg) (No doubts it will be updated to 3.4+)

* ![](https://img.shields.io/badge/pymystem3-0.1.9-yellowgreen.svg)

* ![](https://img.shields.io/badge/Pandas-0.20.3-yellowgreen.svg)

### Optional Service Dependencies
* Named Entity Recognition
    [[flask-python-server]](https://github.com/nicolay-r/ner-flask-wrapper);
    [[sh-script]](start_deep_ner.sh);
* SyntaxNet docker container
    [[sh-script]](contrib/miner/start_syntaxnet.sh);

## Manual

This toolset includes the following instruments and domain-related datasets:

## Installation (Python 2.7)
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

## Research Applications & References

* Attention-Based Neural Networks for Sentiment Attitude Extraction using Distant Supervision 
[[ACM-DOI]](https://doi.org/10.1145/3405962.3405985)
    * Rusnachenko Nicolay, Loukachevitch Natalia
    * WIMS-2020

* Studying Attention Models in Sentiment Attitude Extraction Task 
[[Springer]](https://doi.org/10.1007/978-3-030-51310-8_15) /
[[arXiv:2006.11605]](https://arxiv.org/abs/2006.11605)
    * Rusnachenko Nicolay, Loukachevitch Natalia
    * NLDB-2020

* Distant Supervision for Sentiment Attitude Extraction
[[paper-ranlp-proceeding]](http://lml.bas.bg/ranlp2019/proceedings-ranlp-2019.pdf),
[[poster]](docs/ranlp_2019_poster_portrait.pdf)
    * Rusnachenko Nikolay, Loukachevitch Natalia, Tutubalina Elena
    * RANLP-2019

* Neural Network Approach for Extracting Aggregated Opinions from Analytical Articles 
[[paper]](https://link.springer.com/chapter/10.1007/978-3-030-23584-0_10)
[[code]](https://github.com/nicolay-r/sentiment-pcnn/tree/ccis-2019)
    * Nicolay Rusnachenko, Natalia Loukachevitch 
    * TSD-2018
