* Lemmatization
    - Texterra [[wrapper]](.lemmatization/texterra_wrap.py) -- not supported/utilized in this project;
* Named Entity Recognition (NER) [[API]](ner/base.py):
    - DeepNER [[wrapper]](ner/deepner_wrap.py) -- is a wrapper of IPavlov CRF-BiLSTM model
        [[service]](https://github.com/nicolay-r/ner-bilstm-crf-tensorflow) /
        [[original]](https://github.com/deepmipt/ner);
* Syntax Parser [[API]](syntax/base.py);
    - SyntaxNet [[wrapper]](syntax/syntaxnet_wrap.py);
    - Texterra [[wrapper]](syntax/texterra_wrap.py);
