#!/bin/bash

# Run SyntaxNet docker
docker run --shm-size=1024m -ti --rm -p 8111:9999 inemo/syntaxnet_rus server 0.0.0.0 9999
