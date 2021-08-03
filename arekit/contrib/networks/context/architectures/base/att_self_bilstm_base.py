import tensorflow as tf
from arekit.contrib.networks.attention import common
from arekit.contrib.networks.context.architectures.bilstm import BiLSTM


class AttentionSelfBiLSTMBase(BiLSTM):

    def __init__(self):
        super(AttentionSelfBiLSTMBase, self).__init__()
        self.__att_alphas = None

    def get_attention_output_with_alphas(self, rnn_outputs):
        raise NotImplementedError()

    # region public methods

    def customize_rnn_output(self, rnn_outputs, s_length):
        with tf.variable_scope(common.ATTENTION_SCOPE_NAME):
            att_output, self.__att_alphas = self.get_attention_output_with_alphas(rnn_outputs)

        return att_output

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttentionSelfBiLSTMBase, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield common.ATTENTION_WEIGHTS_LOG_PARAMETER, self.__att_alphas

    # endregion
