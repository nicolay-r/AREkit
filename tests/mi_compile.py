#!/usr/bin/python
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
from core.networks.multi.architectures.mp import MaxPoolingMultiInstanceNetwork
from core.networks.multi.configuration.base import BaseMultiInstanceConfig


def init_config(config):
    assert(isinstance(config, DefaultNetworkConfig))
    config.set_term_embedding(np.zeros((100, 100)))
    config.set_class_weights([1] * config.ClassesCount)
    config.notify_initialization_completed()
    return config


def test_mpmi(context_config, context_network):
    print "TEST: {}".format(context_network)
    config = init_config(BaseMultiInstanceConfig(context_config))
    network = MaxPoolingMultiInstanceNetwork(context_network=context_network)
    network.compile(config, reset_graph=True)


if __name__ == "__main__":

    contexts_supported = [
        (CNNConfig(), VanillaCNN()),
        (CNNConfig(), PiecewiseCNN()),
        (RNNConfig(), RNN()),
        (BiLSTMConfig(), BiLSTM()),
        (RCNNConfig(), RCNN()),
        (IANConfig(), IAN()),
        (AttentionCNNConfig(), AttentionCNN())
    ]

    for config, network in contexts_supported:
        test_mpmi(config, network)
