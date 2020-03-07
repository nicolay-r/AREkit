## AREkit Processing Documentation

* Lemmatization [[API]](lemmatization/base.py);
    - Mystem [[wrapper]](lemmatization/mystem.py) -- Yandex Mystem wrapper
        [[github-repo]](https://github.com/dmitry/yandex_mystem);
    - Texterra [[wrapper]](lemmatization/texterra_wrap.py) -- not supported/utilized in this project;
* Named Entity Recognition (NER) [[API]](ner/base.py):
    - DeepNER [[wrapper]](ner/deepner_wrap.py) -- is a wrapper of IPavlov CRF-BiLSTM model
        [[service]](https://github.com/nicolay-r/ner-flask-wrapper) /
        [[original]](https://github.com/deepmipt/ner);
* Part-Of-Speech Tagging (POS) [[API]](pos/base.py);
    - Mystem [[wrapper]](pos/mystem_wrap.py) -- Yandex Mystem wrapper;
* Syntax Parser [[API]](syntax/base.py);
    - SyntaxNet [[wrapper]](syntax/syntaxnet_wrap.py);
    - Texterra [[wrapper]](syntax/texterra_wrap.py);
* Text Processing
    - Parser [[base-class]](text/parser.py) -- text parser;
    - ParsedText [[base-class]](text/parsed.py) -- processed text;
    - Tokens [[base-class]](text/tokens.py) -- specific text terms, such as: punctuation signs, numbers, URL-links etc.;
