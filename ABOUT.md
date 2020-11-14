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
