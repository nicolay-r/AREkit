#!/bin/bash

# Download and run NER flask wrapper
git clone https://github.com/nicolay-r/ner-flask-wrapper
cd ner-flask-wrapper && nohup python3 start.py &
