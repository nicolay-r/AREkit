#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from core.networks.context.architectures.att_hidden_bilstm import AttentionHiddenBiLSTM
from core.networks.context.architectures.att_ends_cnn import AttentionCNN
from core.networks.context.architectures.bilstm import BiLSTM
from core.networks.context.architectures.cnn import VanillaCNN
from core.networks.context.architectures.att_ends_pcnn import AttentionAttitudeEndsPCNN
from core.networks.context.architectures.contrib.att_frames_bilstm import AttentionFramesBiLSTM
from core.networks.context.architectures.contrib.att_frames_cnn import AttentionFramesCNN
from core.networks.context.architectures.contrib.att_frames_pcnn import AttentionFramesPCNN
from core.networks.context.architectures.contrib.att_hidden_z_yang_bilstm import AttentionHiddenZYangBiLSTM
from core.networks.context.architectures.contrib.ian_ends import IANAttitudeEndsBased
from core.networks.context.architectures.ian_frames import IANFrames
from core.networks.context.architectures.pcnn import PiecewiseCNN
from core.networks.context.architectures.rcnn import RCNN
from core.networks.context.architectures.rnn import RNN
from core.networks.context.architectures.self_att_bilstm import SelfAttentionBiLSTM
from core.networks.context.configurations.att_hidden_bilstm import AttentionHiddenBiLSTMConfig
from core.networks.context.configurations.att_ends_cnn import AttentionAttitudeEndsCNNConfig
from core.networks.context.configurations.base import DefaultNetworkConfig
from core.networks.context.configurations.bi_lstm import BiLSTMConfig
from core.networks.context.configurations.cnn import CNNConfig
from core.networks.context.configurations.att_ends_pcnn import AttentionAttitudeEndsPCNNConfig
from core.networks.context.configurations.contrib.att_frames_bilstm import AttentionFramesBiLSTMConfig
from core.networks.context.configurations.contrib.att_frames_cnn import AttentionFramesCNNConfig
from core.networks.context.configurations.contrib.att_frames_pcnn import AttentionFramesPCNNConfig
from core.networks.context.configurations.contrib.att_hidden_z_yang_bilstm import AttentionHiddenZYangBiLSTMConfig
from core.networks.context.configurations.contrib.ian_ends import IANAttitudeEndsBasedConfig
from core.networks.context.configurations.ian_frames import IANFramesConfig
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
            (AttentionFramesBiLSTMConfig(), AttentionFramesBiLSTM()),
            (AttentionHiddenBiLSTMConfig(), AttentionHiddenBiLSTM()),
            (CNNConfig(), VanillaCNN()),
            (CNNConfig(), PiecewiseCNN()),
            (RNNConfig(), RNN()),
            (BiLSTMConfig(), BiLSTM()),
            (RCNNConfig(), RCNN()),
            (IANFramesConfig(), IANFrames()),

            (IANAttitudeEndsBasedConfig(), IANAttitudeEndsBased()),

            (AttentionAttitudeEndsCNNConfig(), AttentionCNN()),
            (AttentionAttitudeEndsPCNNConfig(), AttentionAttitudeEndsPCNN()),

            (AttentionFramesCNNConfig(), AttentionFramesCNN()),
            (AttentionFramesPCNNConfig(), AttentionFramesPCNN()),

            (AttentionHiddenZYangBiLSTMConfig(), AttentionHiddenZYangBiLSTM())]


if __name__ == "__main__":

    for config, network in contexts_supported():
        print "Compile: {}".format(type(network))
        init_config(config)
        network.compile(config, reset_graph=True)
