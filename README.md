# AREkit

![](logo.png)

**AREkit** -- is a python toolkit for **sentiment attitude extraction** task.

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

* ![](https://img.shields.io/badge/Tensorflow-1.12.0-yellowgreen.svg) 
(`1.6.0+` since `tf.sort` call utilized)

### Optional Service Dependencies
* Named Entity Recognition
    [[flask-python-server]](https://github.com/nicolay-r/ner-flask-wrapper);
    [[sh-script]](start_deep_ner.sh);
* SyntaxNet docker container
    [[sh-script]](start_syntaxnet.sh);

## Manual

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
    * Network [[base-class]](networks/nn.py);
    * Model [[base-class]](networks/tf_model.py);
    * IO [[base-class]](networks/nn_io.py);
    * Callback [[base-class]](networks/callback.py);
    * **Aspect-based Attentive encoders**:
        - Multilayer Perceptron (MLP)
            [[code]](contrib/networks/attention/architectures/mlp.py) /
            [[github:nicolay-r]](https://github.com/nicolay-r/mlp-attention);
    * **Self-based Attentive encoders**:
        - P. Zhou et. al.
            [[code]](contrib/networks/attention/architectures/self_p_zhou.py) /
            [[github:SeoSangwoo]](https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction);
        - Z. Yang et. al.
            [[code]](contrib/networks/attention/architectures/self_z_yang.py) /
            [[github:ilivans]](https://github.com/ilivans/tf-rnn-attention);
    * **Single Sentence Based Architectures**:
        - CNN
            [[code]](contrib/networks/context/architectures/cnn.py) /
            [[github:roomylee]](https://github.com/roomylee/cnn-relation-extraction);
        - CNN + Aspect-based MLP Attention
            [[code]](contrib/networks/context/architectures/base/att_cnn_base.py);
        - PCNN
            [[code]](contrib/networks/context/architectures/pcnn.py) /
            [[github:nicolay-r]](https://github.com/nicolay-r/sentiment-pcnn);
        - PCNN + Aspect-based MLP Attention
            [[code]](contrib/networks/context/architectures/base/att_pcnn_base.py);
        - RNN (LSTM/GRU/RNN)
            [[code]](contrib/networks/context/architectures/rnn.py) /
            [[github:roomylee]](https://github.com/roomylee/rnn-text-classification-tf);
        - IAN (frames based)
            [[code]](contrib/networks/context/architectures/ian_frames.py) /
            [[github:lpq29743]](https://github.com/lpq29743/IAN);
        - RCNN (BiLSTM + CNN)
            [[code]](contrib/networks/context/architectures/rcnn.py) /
            [[github:roomylee]](https://github.com/roomylee/rcnn-text-classification);
        - RCNN + Self Attention
            [[code]](networks/context/architectures/rcnn_self.py);
        - BiLSTM
            [[code]](contrib/networks/context/architectures/bilstm.py) /
            [[github:roomylee]](https://github.com/roomylee/rnn-text-classification-tf);
        - Bi-LSTM + Aspect-based MLP Attention 
            [[code]](contrib/networks/context/architectures/base/att_bilstm_base.py)
        - Bi-LSTM + Self Attention
            [[code]](contrib/networks/context/architectures/self_att_bilstm.py) /
            [[github:roomylee]](https://github.com/roomylee/self-attentive-emb-tf);
        - RCNN + Self Attention
            [[code]](contrib/networks/context/architectures/att_self_rcnn.py);
    * **Multi Sentence Based Encoders Architectures**:
        - Self Attentive 
            [[code]](contrib/networks/multi/architectures/att_self.py);
        - Max Pooling
            [[code]](contrib/networks/multi/architectures/max_pooling.py) /
            [[paper]](https://pdfs.semanticscholar.org/8731/369a707046f3f8dd463d1fd107de31d40a24.pdf);
        - Single MLP
            [[code]](networks/multi/architectures/base_single_mlp.py);
* **Sources** [[README]](source/README.md) -- datasets and embeddings;
    * RuAttitudes [[github-repo]](https://github.com/nicolay-r/RuAttitudes);
    * RuSentiFrames [[github-repo]](https://github.com/nicolay-r/RuSentiFrames);
    * RuSentRel [[github-repo]](https://github.com/nicolay-r/RuSentRel);
    * Embeddings
        * RusVectores 
            [[code]](source/embeddings/rusvectores.py) /
            [[news-w2v-download]](http://rusvectores.org/static/models/rusvectores2/news_mystem_skipgram_1000_20_2015.bin.gz);
    * Lexicons
        * RuSentiLex [[lab-site]](https://www.labinform.ru/pub/rusentilex/index.htm);
* **Evaluation** -- tools that allows to perform models quality assessment.
    * CmpOpinion [[API]](evaluation/cmp_opinions.py) -- structure describes pairs of opinions to compare;
    * BaseEvaluator [[API]](evaluation/evaluators/base.py);

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

## Might be embedded in framework:

* Semantic Relation Classification via Hierarchical Recurrent Neural Network with Attention
[[paper]](https://www.aclweb.org/anthology/C16-1119)

## Research Applications & References

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
