from arekit.contrib.networks.tf_helpers.cell_types import CellTypes
from arekit.contrib.networks.context.configurations.rcnn import RCNNConfig


def ctx_rcnn_z_yang_custom_config(config):
    assert(isinstance(config, RCNNConfig))
    config.modify_cell_type(CellTypes.LSTM)
    config.modify_dropout_rnn_keep_prob(0.9)
