import tensorflow as tf
from arekit.contrib.networks.context.configurations.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTMConfig


def ctx_att_bilstm_z_yang_custom_config(config):
    assert(isinstance(config, AttentionSelfZYangBiLSTMConfig))
    config.modify_weight_initializer(tf.contrib.layers.xavier_initializer())
