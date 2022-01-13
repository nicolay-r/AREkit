import tensorflow as tf

from arekit.contrib.networks.context.configurations.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTMConfig
from arekit.contrib.networks.context.configurations.att_self_z_yang_bilstm import AttentionSelfZYangBiLSTMConfig
from arekit.contrib.networks.context.configurations.bilstm import BiLSTMConfig
from arekit.contrib.networks.context.configurations.cnn import CNNConfig
from arekit.contrib.networks.context.configurations.rcnn import RCNNConfig
from arekit.contrib.networks.context.configurations.rnn import RNNConfig
from arekit.contrib.networks.context.configurations.self_att_bilstm import SelfAttentionBiLSTMConfig
from arekit.contrib.networks.tf_helpers.cell_types import CellTypes
from examples.network.args.const import TERMS_PER_CONTEXT


def ctx_self_att_bilstm_custom_config(config):
    assert(isinstance(config, SelfAttentionBiLSTMConfig))
    config.modify_penaltization_term_coef(0.5)
    config.modify_cell_type(CellTypes.BasicLSTM)
    config.modify_dropout_rnn_keep_prob(0.8)
    config.modify_terms_per_context(TERMS_PER_CONTEXT)


def ctx_att_bilstm_p_zhou_custom_config(config):
    assert(isinstance(config, AttentionSelfPZhouBiLSTMConfig))
    config.modify_hidden_size(128)
    config.modify_cell_type(CellTypes.LSTM)
    config.modify_dropout_rnn_keep_prob(0.9)


def ctx_att_bilstm_z_yang_custom_config(config):
    assert(isinstance(config, AttentionSelfZYangBiLSTMConfig))
    config.modify_weight_initializer(tf.contrib.layers.xavier_initializer())


def ctx_bilstm_custom_config(config):
    assert(isinstance(config, BiLSTMConfig))
    config.modify_hidden_size(128)
    config.modify_cell_type(CellTypes.BasicLSTM)
    config.modify_dropout_rnn_keep_prob(0.8)
    config.modify_terms_per_context(TERMS_PER_CONTEXT)


def ctx_cnn_custom_config(config):
    assert(isinstance(config, CNNConfig))
    config.modify_weight_initializer(tf.contrib.layers.xavier_initializer())


def ctx_lstm_custom_config(config):
    assert(isinstance(config, RNNConfig))
    config.modify_cell_type(CellTypes.BasicLSTM)
    config.modify_hidden_size(128)
    config.modify_dropout_rnn_keep_prob(0.8)
    config.modify_terms_per_context(TERMS_PER_CONTEXT)


def ctx_pcnn_custom_config(config):
    assert(isinstance(config, CNNConfig))
    config.modify_weight_initializer(tf.contrib.layers.xavier_initializer())


def ctx_rcnn_custom_config(config):
    assert(isinstance(config, RCNNConfig))
    config.modify_cell_type(CellTypes.LSTM)
    config.modify_dropout_rnn_keep_prob(0.9)


def ctx_rcnn_p_zhou_custom_config(config):
    assert(isinstance(config, RCNNConfig))
    config.modify_cell_type(CellTypes.LSTM)
    config.modify_dropout_rnn_keep_prob(0.9)


def ctx_rcnn_z_yang_custom_config(config):
    assert(isinstance(config, RCNNConfig))
    config.modify_cell_type(CellTypes.LSTM)
    config.modify_dropout_rnn_keep_prob(0.9)