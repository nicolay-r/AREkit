#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from core.networks.context.architectures.att_bilstm import AttBiLSTM
from core.networks.context.architectures.att_cnn import AttentionCNN
from core.networks.context.architectures.bi_lstm import BiLSTM
from core.networks.context.architectures.cnn import VanillaCNN
from core.networks.context.architectures.contrib.att_ends_pcnn import AttentionPCNN
from core.networks.context.architectures.contrib.ian_ends import IANAttituteEndsBased
from core.networks.context.architectures.ian import IAN
from core.networks.context.architectures.pcnn import PiecewiseCNN
from core.networks.context.architectures.rcnn import RCNN
from core.networks.context.architectures.rnn import RNN
from core.networks.context.architectures.self_att_bilstm import SelfAttentionBiLSTM
from core.networks.context.configurations.att_bilstm import AttBiLSTMConfig
from core.networks.context.configurations.att_cnn import AttentionCNNConfig
from core.networks.context.configurations.base import DefaultNetworkConfig
from core.networks.context.configurations.bi_lstm import BiLSTMConfig
from core.networks.context.configurations.cnn import CNNConfig
from core.networks.context.configurations.ian import IANConfig
from core.networks.context.configurations.rcnn import RCNNConfig
from core.networks.context.configurations.rnn import RNNConfig
from core.networks.context.configurations.self_att_bilstm import SelfAttentionBiLSTMConfig


def init_config(config):
    assert(isinstance(config, DefaultNetworkConfig))
    config.set_term_embedding(np.zeros((100, 100)))
    config.set_class_weights([1] * config.ClassesCount)
    config.notify_initialization_completed()


def contexts_supported():
    return [(SelfAttentionBiLSTMConfig(), SelfAttentionBiLSTM()),
            (AttBiLSTMConfig(), AttBiLSTM()),
            (CNNConfig(), VanillaCNN()),
            (CNNConfig(), PiecewiseCNN()),
            (RNNConfig(), RNN()),
            (BiLSTMConfig(), BiLSTM()),
            (RCNNConfig(), RCNN()),
            (IANConfig(), IAN()),
            (IANConfig(), IANAttituteEndsBased()),
            (AttentionCNNConfig(), AttentionCNN()),
            (AttentionCNNConfig(), AttentionPCNN())]


if __name__ == "__main__":

    for config, network in contexts_supported():
        print "Compile: {}".format(type(network))
        init_config(config)
        network.compile(config, reset_graph=True)
