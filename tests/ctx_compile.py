#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys

sys.path.append('../../../')
import numpy as np
from networks.context.configurations.base import CommonModelSettings
from networks.context.architectures.cnn import VanillaCNN
from networks.context.configurations.cnn import CNNConfig
from networks.context.architectures.pcnn import PiecewiseCNN
from networks.context.configurations.rnn import RNNConfig
from networks.context.architectures.rnn import RNN
from networks.context.architectures.bi_lstm import BiLSTM
from networks.context.configurations.bi_lstm import BiLSTMConfig
from networks.context.architectures.ian import IAN
from networks.context.configurations.ian import IANConfig
from networks.context.architectures.rcnn import RCNN
from networks.context.configurations.rcnn import RCNNConfig


def init_config(config):
    assert(isinstance(config, CommonModelSettings))
    config.set_term_embedding(np.zeros((100, 100)))
    config.set_class_weights([1] * config.ClassesCount)
    return config


def test_cnn():
    print "Loading CNN Configuration ..."
    config = init_config(CNNConfig())
    cnn = VanillaCNN()
    pcnn = PiecewiseCNN()
    cnn.compile(config)
    pcnn.compile(config)

def test_rnn():
    config = init_config(RNNConfig())
    rnn = RNN()
    rnn.compile(config)

def test_rcnn():
    config = init_config(RCNNConfig())
    rcnn = RCNN()
    rcnn.compile(config)

def bilstm():
    config = init_config(BiLSTMConfig())
    bilstm = BiLSTM()
    bilstm.compile(config)


def test_ian():
    config = init_config(IANConfig())
    arnn = IAN()
    arnn.compile(config)


if __name__ == "__main__":

    test_cnn()
    test_rnn()
