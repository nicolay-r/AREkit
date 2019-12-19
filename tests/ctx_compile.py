#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging

import numpy as np

from arekit.networks.context.architectures.contrib.att_ef_bilstm import AttentionEndsAndFramesBiLSTM
from arekit.networks.context.architectures.contrib.att_ef_cnn import AttentionEndsAndFramesCNN
from arekit.networks.context.architectures.contrib.att_ef_pcnn import AttentionEndsAndFramesPCNN
from arekit.networks.context.architectures.contrib.att_hidden_p_zhou_bilstm import AttentionHiddenPZhouBiLSTM
from arekit.networks.context.architectures.contrib.att_ends_cnn import AttentionEndsCNN
from arekit.networks.context.architectures.bilstm import BiLSTM
from arekit.networks.context.architectures.cnn import VanillaCNN
from arekit.networks.context.architectures.contrib.att_ends_pcnn import AttentionEndsPCNN
from arekit.networks.context.architectures.contrib.att_frames_bilstm import AttentionFramesBiLSTM
from arekit.networks.context.architectures.contrib.att_frames_cnn import AttentionFramesCNN
from arekit.networks.context.architectures.contrib.att_frames_pcnn import AttentionFramesPCNN
from arekit.networks.context.architectures.contrib.att_hidden_z_yang_bilstm import AttentionHiddenZYangBiLSTM
from arekit.networks.context.architectures.contrib.att_syn_ends_bilstm import AttentionSynonymEndsBiLSTM
from arekit.networks.context.architectures.contrib.att_syn_ends_cnn import AttentionSynonymEndsCNN
from arekit.networks.context.architectures.contrib.att_syn_ends_pcnn import AttentionSynonymEndsPCNN
from arekit.networks.context.architectures.contrib.ian_ef import IANEndsAndFramesBased
from arekit.networks.context.architectures.contrib.ian_ends import IANEndsBased
from arekit.networks.context.architectures.contrib.ian_syn_ends import IANSynonymEndsBased
from arekit.networks.context.architectures.contrib.ian_frames import IANFrames
from arekit.networks.context.architectures.pcnn import PiecewiseCNN
from arekit.networks.context.architectures.rcnn import RCNN
from arekit.networks.context.architectures.rnn import RNN
from arekit.networks.context.architectures.self_att_bilstm import SelfAttentionBiLSTM
from arekit.networks.context.configurations.contrib.att_ef_bilstm import AttentionEndsAndFramesBiLSTMConfig
from arekit.networks.context.configurations.contrib.att_ef_cnn import AttentionEndsAndFramesCNNConfig
from arekit.networks.context.configurations.contrib.att_ef_pcnn import AttentionEndsAndFramesPCNNConfig
from arekit.networks.context.configurations.contrib.att_hidden_p_zhou_bilstm import AttentionHiddenPZhouBiLSTMConfig
from arekit.networks.context.configurations.contrib.att_ends_cnn import AttentionEndsCNNConfig
from arekit.networks.context.configurations.base import DefaultNetworkConfig
from arekit.networks.context.configurations.bilstm import BiLSTMConfig
from arekit.networks.context.configurations.cnn import CNNConfig
from arekit.networks.context.configurations.contrib.att_ends_pcnn import AttentionEndsPCNNConfig
from arekit.networks.context.configurations.contrib.att_frames_bilstm import AttentionFramesBiLSTMConfig
from arekit.networks.context.configurations.contrib.att_frames_cnn import AttentionFramesCNNConfig
from arekit.networks.context.configurations.contrib.att_frames_pcnn import AttentionFramesPCNNConfig
from arekit.networks.context.configurations.contrib.att_hidden_z_yang_bilstm import AttentionHiddenZYangBiLSTMConfig
from arekit.networks.context.configurations.contrib.att_syn_ends_bilstm import AttentionSynonymEndsBiLSTMConfig
from arekit.networks.context.configurations.contrib.att_syn_ends_cnn import AttentionSynonymEndsCNNConfig
from arekit.networks.context.configurations.contrib.att_syn_ends_pcnn import AttentionSynonymEndsPCNNConfig
from arekit.networks.context.configurations.contrib.ian_ef import IANEndsAndFramesConfig
from arekit.networks.context.configurations.contrib.ian_ends import IANEndsBasedConfig
from arekit.networks.context.configurations.contrib.ian_syn_ends import IANSynonymEndsBasedConfig
from arekit.networks.context.configurations.contrib.ian_frames import IANFramesConfig
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

            (AttentionEndsAndFramesBiLSTMConfig(), AttentionEndsAndFramesBiLSTM()),
            (AttentionFramesBiLSTMConfig(), AttentionFramesBiLSTM()),

            (AttentionSynonymEndsBiLSTMConfig(), AttentionSynonymEndsBiLSTM()),
            (AttentionHiddenPZhouBiLSTMConfig(), AttentionHiddenPZhouBiLSTM()),
            (AttentionHiddenZYangBiLSTMConfig(), AttentionHiddenZYangBiLSTM()),

            (CNNConfig(), VanillaCNN()),
            (CNNConfig(), PiecewiseCNN()),
            (RNNConfig(), RNN()),
            (BiLSTMConfig(), BiLSTM()),
            (RCNNConfig(), RCNN()),

            (IANFramesConfig(), IANFrames()),
            # (IANEndsAndFramesConfig(), IANEndsAndFramesBased()),
            (IANEndsBasedConfig(), IANEndsBased()),
            (IANSynonymEndsBasedConfig(), IANSynonymEndsBased()),

            (AttentionEndsAndFramesPCNNConfig(), AttentionEndsAndFramesPCNN()),
            (AttentionEndsAndFramesCNNConfig(), AttentionEndsAndFramesCNN()),
            (AttentionEndsCNNConfig(), AttentionEndsCNN()),
            (AttentionEndsPCNNConfig(), AttentionEndsPCNN()),
            (AttentionSynonymEndsPCNNConfig(), AttentionSynonymEndsPCNN()),
            (AttentionSynonymEndsCNNConfig(), AttentionSynonymEndsCNN()),

            (AttentionFramesCNNConfig(), AttentionFramesCNN()),
            (AttentionFramesPCNNConfig(), AttentionFramesPCNN()),
            ]


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    for config, network in contexts_supported():
        logger.info("Compile: {}".format(type(network)))
        init_config(config)
        network.compile(config, reset_graph=True)
