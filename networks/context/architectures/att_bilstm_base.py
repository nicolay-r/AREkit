import tensorflow as tf

from arekit.networks.attention.helpers import embedding
from arekit.networks.attention import common
from arekit.networks.context.architectures.bilstm import BiLSTM
from arekit.networks.tf_helpers import sequence


class AttentionBiLSTMBase(BiLSTM):

    def __init__(self):
        super(AttentionBiLSTMBase, self).__init__()
        self.__att_alphas = None

    # region properties

    @property
    def ContextEmbeddingSize(self):
        return super(AttentionBiLSTMBase, self).ContextEmbeddingSize + \
               self.Config.AttentionModel.AttentionEmbeddingSize

    # endregion

    def get_att_input(self):
        """
        This is an abstract method which is considered to be implemented in nested class.
        """
        raise NotImplementedError()

    # region public 'init' methods

    def init_hidden_states(self):
        super(AttentionBiLSTMBase, self).init_hidden_states()
        with tf.variable_scope(common.ATTENTION_SCOPE_NAME):
            self.Config.AttentionModel.init_hidden()

    def init_input(self):
        super(AttentionBiLSTMBase, self).init_input()
        with tf.variable_scope(common.ATTENTION_SCOPE_NAME):
            self.Config.AttentionModel.init_input(p_names_with_sizes=embedding.get_ns(self))

    # endregion

    def customize_rnn_output(self, rnn_outputs, s_length):

        g = sequence.select_last_relevant_in_sequence(rnn_outputs, s_length)

        with tf.variable_scope(common.ATTENTION_SCOPE_NAME):
            att_e, self.__att_alphas = embedding.init_mlp_attention_embedding(
                ctx_network=self,
                mlp_att=self.Config.AttentionModel,
                keys=self.get_att_input())

        return tf.concat([g, att_e], axis=-1)

    # region public 'iter' methods

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttentionBiLSTMBase, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield common.ATTENTION_WEIGHTS_LOG_PARAMETER, self.__att_alphas

    # endregion
