#!/bin/bash

# Performing a quick library installation.
# https://stackoverflow.com/questions/19048732/python-setup-py-develop-vs-install
sudo pip install -e ../ --no-deps

# Run all unit tests.
python2.7 -m unittest discover .
