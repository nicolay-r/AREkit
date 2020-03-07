## AREkit Processing Documentation

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
