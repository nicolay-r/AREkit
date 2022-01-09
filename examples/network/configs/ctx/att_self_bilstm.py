from arekit.contrib.networks.tf_helpers.cell_types import CellTypes
from arekit.contrib.networks.context.configurations.self_att_bilstm import SelfAttentionBiLSTMConfig
from examples.network.args.default import TERMS_PER_CONTEXT


def ctx_self_att_bilstm_custom_config(config):
    assert(isinstance(config, SelfAttentionBiLSTMConfig))
    config.modify_penaltization_term_coef(0.5)
    config.modify_cell_type(CellTypes.BasicLSTM)
    config.modify_dropout_rnn_keep_prob(0.8)
    config.modify_terms_per_context(TERMS_PER_CONTEXT)
