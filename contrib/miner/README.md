## Processing / Miner library

* Lemmatization
    - Texterra [[wrapper]](.lemmatization/texterra_wrap.py) -- not supported/utilized in this project;
* Named Entity Recognition (NER) [[API]](ner/base.py):
    - DeepNER [[wrapper]](ner/deepner_wrap.py) -- is a wrapper of IPavlov CRF-BiLSTM model
        [[service]](https://github.com/nicolay-r/ner-bilstm-crf-tensorflow) /
        [[original]](https://github.com/deepmipt/ner);
* Syntax Parser [[API]](syntax/base.py);
    - SyntaxNet [[wrapper]](syntax/syntaxnet_wrap.py);
    - Texterra [[wrapper]](syntax/texterra_wrap.py);
    
### Optional Service Dependencies
* Named Entity Recognition
    [[flask-python-server]](https://github.com/nicolay-r/ner-flask-wrapper);
    [[sh-script]](start_deep_ner.sh);
* SyntaxNet docker container
    [[sh-script]](contrib/miner/start_syntaxnet.sh);

