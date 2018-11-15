#!/bin/bash

# Download and run NER flask wrapper
git clone https://github.com/nicolay-r/ner-flask-wrapper
cd ner-flask-wrapper && nohup python3 start.py &

# Run SyntaxNet docker
docker run --shm-size=1024m -ti --rm -p 8111:9999 inemo/syntaxnet_rus server 0.0.0.0 9999
