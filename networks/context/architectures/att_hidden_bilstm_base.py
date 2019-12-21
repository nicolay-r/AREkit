import tensorflow as tf
from arekit.networks.attention import common
from arekit.networks.context.architectures.bilstm import BiLSTM


class AttentionHiddenBiLSTMBase(BiLSTM):
    """
    NOTE: We consider 'hidden' since attention utilize a hidden states as keys towards
    embedded context.
    """

    def __init__(self):
        super(AttentionHiddenBiLSTMBase, self).__init__()
        self.__att_alphas = None

    def get_attention_output_with_alphas(self, rnn_outputs):
        raise NotImplementedError()

    # region public methods

    def customize_rnn_output(self, rnn_outputs, s_length):
        with tf.variable_scope(common.ATTENTION_SCOPE_NAME):
            att_output, self.__att_alphas = self.get_attention_output_with_alphas(rnn_outputs)

        return att_output

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttentionHiddenBiLSTMBase, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield common.ATTENTION_WEIGHTS_LOG_PARAMETER, self.__att_alphas

    # endregion
