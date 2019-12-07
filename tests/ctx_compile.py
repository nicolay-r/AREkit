#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from arekit.networks.context.architectures.att_hidden_bilstm import AttentionHiddenBiLSTM
from arekit.networks.context.architectures.att_ends_cnn import AttentionCNN
from arekit.networks.context.architectures.bilstm import BiLSTM
from arekit.networks.context.architectures.cnn import VanillaCNN
from arekit.networks.context.architectures.att_ends_pcnn import AttentionAttitudeEndsPCNN
from arekit.networks.context.architectures.contrib.att_frames_bilstm import AttentionFramesBiLSTM
from arekit.networks.context.architectures.contrib.att_frames_cnn import AttentionFramesCNN
from arekit.networks.context.architectures.contrib.att_frames_pcnn import AttentionFramesPCNN
from arekit.networks.context.architectures.contrib.att_hidden_z_yang_bilstm import AttentionHiddenZYangBiLSTM
from arekit.networks.context.architectures.contrib.ian_ends import IANAttitudeEndsBased
from arekit.networks.context.architectures.ian_frames import IANFrames
from arekit.networks.context.architectures.pcnn import PiecewiseCNN
from arekit.networks.context.architectures.rcnn import RCNN
from arekit.networks.context.architectures.rnn import RNN
from arekit.networks.context.architectures.self_att_bilstm import SelfAttentionBiLSTM
from arekit.networks.context.configurations.att_hidden_bilstm import AttentionHiddenBiLSTMConfig
from arekit.networks.context.configurations.att_ends_cnn import AttentionAttitudeEndsCNNConfig
from arekit.networks.context.configurations.base import DefaultNetworkConfig
from arekit.networks.context.configurations.bilstm import BiLSTMConfig
from arekit.networks.context.configurations.cnn import CNNConfig
from arekit.networks.context.configurations.att_ends_pcnn import AttentionAttitudeEndsPCNNConfig
from arekit.networks.context.configurations.contrib.att_frames_bilstm import AttentionFramesBiLSTMConfig
from arekit.networks.context.configurations.contrib.att_frames_cnn import AttentionFramesCNNConfig
from arekit.networks.context.configurations.contrib.att_frames_pcnn import AttentionFramesPCNNConfig
from arekit.networks.context.configurations.contrib.att_hidden_z_yang_bilstm import AttentionHiddenZYangBiLSTMConfig
from arekit.networks.context.configurations.contrib.ian_ends import IANAttitudeEndsBasedConfig
from arekit.networks.context.configurations.ian_frames import IANFramesConfig
from arekit.networks.context.configurations.rcnn import RCNNConfig
from arekit.networks.context.configurations.rnn import RNNConfig
from arekit.networks.context.configurations.self_att_bilstm import SelfAttentionBiLSTMConfig


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
