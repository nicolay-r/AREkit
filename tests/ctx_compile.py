#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from core.networks.context.architectures.att_cnn import AttentionCNN
from core.networks.context.architectures.bi_lstm import BiLSTM
from core.networks.context.architectures.cnn import VanillaCNN
from core.networks.context.architectures.ian import IAN
from core.networks.context.architectures.pcnn import PiecewiseCNN
from core.networks.context.architectures.rcnn import RCNN
from core.networks.context.architectures.rnn import RNN
from core.networks.context.configurations.att_cnn import AttentionCNNConfig
from core.networks.context.configurations.base import DefaultNetworkConfig
from core.networks.context.configurations.bi_lstm import BiLSTMConfig
from core.networks.context.configurations.cnn import CNNConfig
from core.networks.context.configurations.ian import IANConfig
from core.networks.context.configurations.rcnn import RCNNConfig
from core.networks.context.configurations.rnn import RNNConfig


def init_config(config):
    assert(isinstance(config, DefaultNetworkConfig))
    config.set_term_embedding(np.zeros((100, 100)))
    config.set_class_weights([1] * config.ClassesCount)
    config.notify_initialization_completed()
    return config


def test_cnn_pcnn():
    config = init_config(CNNConfig())
    cnn = VanillaCNN()
    pcnn = PiecewiseCNN()
    cnn.compile(config, reset_graph=True)
    pcnn.compile(config, reset_graph=True)


def test_rnn():
    config = init_config(RNNConfig())
    rnn = RNN()
    rnn.compile(config, reset_graph=True)


def test_rcnn():
    config = init_config(RCNNConfig())
    rcnn = RCNN()
    rcnn.compile(config, reset_graph=True)


def test_bilstm():
    config = init_config(BiLSTMConfig())
    bilstm = BiLSTM()
    bilstm.compile(config, reset_graph=True)


def test_ian():
    config = init_config(IANConfig())
    arnn = IAN()
    arnn.compile(config, reset_graph=True)

def test_attcnn():
    config = init_config(AttentionCNNConfig())
    attcnn = AttentionCNN()
    attcnn.compile(config, reset_graph=True)


if __name__ == "__main__":

    # CNN models
    test_cnn_pcnn()

    # Recurrent networks
    test_rnn()
    test_rcnn()
    test_bilstm()

    # Models with attention
    test_ian()
    test_attcnn()
