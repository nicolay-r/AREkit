#!/bin/bash

#
# TODO. This should be removed.
#

# Download and run NER flask wrapper
git clone https://github.com/nicolay-r/ner-bilstm-crf-tensorflow
cd ner-flask-wrapper && nohup python3 start.py &
