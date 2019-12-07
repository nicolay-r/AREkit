from arekit.networks.attention.architectures.rnn_attention_z_yang import attention_by_z_yang
from arekit.networks.context.architectures.att_hidden_bilstm import AttentionHiddenBiLSTM
from arekit.networks.context.configurations.contrib.att_hidden_z_yang_bilstm import AttentionHiddenZYangBiLSTMConfig


class AttentionHiddenZYangBiLSTM(AttentionHiddenBiLSTM):

    def get_attention_output_with_alphas(self, rnn_outputs):
        assert(isinstance(self.Config, AttentionHiddenZYangBiLSTMConfig))
        return attention_by_z_yang(inputs=rnn_outputs,
                                   attention_size=self.Config.AttentionSize,
                                   return_alphas=True)