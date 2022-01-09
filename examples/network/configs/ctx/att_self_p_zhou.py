from arekit.contrib.networks.tf_helpers.cell_types import CellTypes
from arekit.contrib.networks.context.configurations.att_self_p_zhou_bilstm import AttentionSelfPZhouBiLSTMConfig


def ctx_att_bilstm_p_zhou_custom_config(config):
    assert(isinstance(config, AttentionSelfPZhouBiLSTMConfig))
    config.modify_hidden_size(128)
    config.modify_cell_type(CellTypes.LSTM)
    config.modify_dropout_rnn_keep_prob(0.9)
