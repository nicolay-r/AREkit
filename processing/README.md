## AREkit Processing Documentation

* Lemmatization [[API]](lemmatization/base.py);
    - Mystem [[wrapper]](lemmatization/mystem.py) -- Yandex Mystem wrapper
        [[github-repo]](https://github.com/dmitry/yandex_mystem);
    - Texterra [[wrapper]](lemmatization/texterra_wrap.py) -- not supported/utilized in this project;
* Part-Of-Speech Tagging (POS) [[API]](pos/base.py);
    - Mystem [[wrapper]](pos/mystem_wrap.py) -- Yandex Mystem wrapper;
* Text Processing
    - Parser [[base-class]](text/parser.py) -- text parser;
    - ParsedText [[base-class]](text/parsed.py) -- processed text;
    - Tokens [[base-class]](text/tokens.py) -- specific text terms, such as: punctuation signs, numbers, URL-links etc.;
