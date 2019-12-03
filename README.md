# AREkit

![](logo.png)

> Note: Nowadays it utilize `core` directory instead of `arekit`.

**AREkit** -- is a python toolkit (library) for **sentiment attitudes extraction** (specific relation extraction) task.

# Contents
* [Description](#description)
* [Dependencies](#dependencies)
* [Manual](#manual-update-in-progress)
* [Installation](#installation)
* [How to use examples](#usage-update-in-progress)

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

### Optional Service Dependencies
* Named Entity Recognition
    [[flask-python-server]](https://github.com/nicolay-r/ner-flask-wrapper);
* SyntaxNet docker container
    [[github-repo]]() /
    [[sh-script]]();

## Manual [Update in Progress]

This toolset includes the following instruments and domain-related datasets:

* **Common** [[DIR]](networks) -- fundamential structures and types utilized in Sentiment Attitudes Extraction Task;
    * Bound [[base-class]](common/bound.py) -- range in text;
    * TextObject [[base-class]](common/text_object.py) -- any entry in text with related *Bound*;
    * Entity [[base-class]](common/entities/base.py) -- same as TextObject but related to specific text entries;
    * Opinion [[base-class]](common/opinions/base.py) -- actually text attitudes with 'source' and 'destination' ('X' -> 'Y');
    * Label [[base-classes]](common/labels/base.py) -- sentiment label;
    * Frame;
    * FrameVariant [[base-class]](common/frame_variants/base.py);
    * Embedding [[base-class]](common/embeddings/base.py) -- base class for Word2Vec-like embeddings;
    * Synonyms [[base-class]](common/synonyms.py) -- storage for synonymous entries (words and phrases);
* **Processing** [[README]](processing/README.md);
    * Lemmatization [[API]](processing/lemmatization/base.py);
        - Mystem [[wrapper]](processing/lemmatization/mystem.py) -- Yandex Mystem wrapper
            [[github-repo]](https://github.com/dmitry/yandex_mystem);
        - Texterra [[wrapper]](processing/lemmatization/texterra_wrap.py) -- not supported/utilized in this project;
    * Named Entity Recognition (NER) [[API]](processing/ner/base.py):
        - DeepNER [[wrapper]](processing/ner/deepner_wrap.py) -- is a wrapper of IPavlov CRF-BiLSTM model
            [[service]](https://github.com/nicolay-r/ner-flask-wrapper) /
            [[original]](https://github.com/deepmipt/ner);
    * Part-Of-Speech Tagging (POS) [[API]](processing/pos/base.py);
        - Mystem [[wrapper]](processing/pos/mystem_wrap.py) -- Yandex Mystem wrapper;
    * Syntax Parser [[API]](processing/syntax/base.py);
        - SyntaxNet [[wrapper]](processing/syntax/syntaxnet_wrap.py);
        - Texterra [[wrapper]](processing/syntax/texterra_wrap.py);
    * Text Processing
        - Parser [[base-class]](processing/text/parser.py) -- text parser;
        - ParsedText [[base-class]](processing/text/parsed.py) -- processed text;
        - Tokens [[base-class]](processing/text/tokens.py) -- specific text terms, such as: punctuation signs, numbers, URL-links etc.;
* **Neural Networks** [[README]](networks/README.md)
    ![](https://img.shields.io/badge/Tensorflow-1.14.0-yellowgreen.svg)
    * Network [[base-class]]();
    * Model [[base-class]]();
    * IO [[base-class]]();
    * Callback [[base-class]]();
    * **Attention Architectures**:
        - Multilayer Perceptron (MLP)
            [[code]]() /
            [[github]]();
        - P. Zhou, RNN-output based
            [[code]]() /
            [[github]]();
        - Z. Yang, RNN-output based
            [[code]]() /
            [[github]]();
    * **Single Sentence Based Architectures** [[base]]();
        - CNN
            [[code]]() /
            [[github]]();
        - PCNN
            [[code]]() /
            [[github]]();
        - Attention-CNN
            [[code]]() /
            [[github]]();
        - Attention-PCNN
            [[code]]();
        - RNN (LSTM/GRU/RNN)
            [[code]]() /
            [[github]]();
        - IAN
            [[code]]() /
            [[github]]();
        - RCNN (RCNN)
            [[code]]() /
            [[github]]();
        - BiLSTM
            [[code]]() /
            [[github]]();
        - Self Attention Bi-LSTM
            [[code]]() /
            [[github]]();
    * **Multi Sentence Based Architectures** [[base]]();
        - Attention Hidden
            [[code]]() /
            [[github]]();
        - Attention MLP Frames
            [[code]]() /
            [[github]]();
        - Max Pooling
            [[code]]() /
            [[github]]();
        - Single MLP
            [[code]]() /
            [[github]]();
* **Sources** [[README]](source/README.md) -- datasets and embeddings;
    * RuAttitudes [[github-repo]](https://github.com/nicolay-r/RuAttitudes);
    * RuSentiFrames [[github-repo]](https://github.com/nicolay-r/RuSentiFrames);
    * RuSentRel [[github-repo]](https://github.com/nicolay-r/RuSentRel);
* **Evaluation** -- tools that allows to perform models quality assessment.
    * CmpOpinion [[API]](evaluation/cmp_opinions.py) -- structure describes pairs of opinions to compare;
    * BaseEvaluator [[API]](evaluation/evaluators/base.py);

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

