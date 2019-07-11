#!/usr/bin/python
import sys

sys.path.append('../../../')
import numpy as np
from networks.context.architectures.bi_lstm import BiLSTM
from networks.context.architectures.rnn import RNN
from networks.context.configurations.bi_lstm import BiLSTMConfig
from networks.context.configurations.rnn import RNNConfig
from networks.context.configurations.base import CommonModelSettings
from networks.context.architectures.pcnn import PiecewiseCNN
from networks.context.architectures.cnn import VanillaCNN
from networks.mimlre.base import MIMLRE
from networks.context.configurations.cnn import CNNConfig
from networks.mimlre.configuration.base import MIMLRESettings


def init_config(config):
    assert(isinstance(config, CommonModelSettings))
    config.set_term_embedding(np.zeros((100, 100)))
    config.set_class_weights([1, 1, 1])
    return config


def compile(context_config, context_network):
    cfg = init_config(MIMLRESettings(context_config))
    network = MIMLRE(context_network=context_network)
    network.compile(cfg)


if __name__ == "__main__":

    ctx = [
        # (CNNConfig(), VanillaCNN()),
        # (CNNConfig(), PiecewiseCNN()),
        (RNNConfig(), RNN()),
        (BiLSTMConfig(), BiLSTM())
    ]

    for config, network in ctx:
        compile(config, network)

