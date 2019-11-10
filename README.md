# AREkit

> Note: Nowadays it utilize `core` directory instead of `arekit`. 

**saekit** -- is a python toolkit (library) for **sentiment attitudes extraction** task.

#### Table of Contents
1. Description.
2. Dependencies.
3. Manual.
4. Installation.
5. How to use examples.

## Description

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

## Dependencies

![](https://img.shields.io/badge/Python-2.7-brightgreen.svg)
![](https://img.shields.io/badge/Tensorflow-1.14.0-yellowgreen.svg)

List of the toolset dependencies are as follows:
* Python 2.7 (No doubts it will be updated to 3.4+)
* pymystem3==0.1.9
* pandas==0.20.3
* tensorflow==1.4.1

## Manual [Update in Progress]

This toolset includes the following instruments and domain-related datasets:

* **Common** [[DIR]](networks) -- fundamential structures and types utilized in Sentiment Attitudes Extraction Task;
    * Embedding [[API]](common/embedding.py);
* **Processing** [[README]](processing/README.md);
* **Neural Networks** [[README]](networks/README.md);
* **Sources** [[README]](source/README.md) -- datasets and embeddings;
* **Evaluation** -- tools that allows to perform models quality assessment. 


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

## Usage [Update in Progress]

> TODO: Provide list of related projects.

