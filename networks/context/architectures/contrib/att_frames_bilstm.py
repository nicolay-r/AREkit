import tensorflow as tf

from arekit.networks.attention.helpers import embedding
from arekit.networks.context.architectures.bilstm import BiLSTM
from arekit.networks.context.sample import InputSample
from arekit.networks.tf_helpers import sequence


class AttentionFramesBiLSTM(BiLSTM):

    __attention_scope = 'mlp-attention-model'

    def __init__(self):
        super(AttentionFramesBiLSTM, self).__init__()
        self.__att_alphas = None

    # region properties

    @property
    def ContextEmbeddingSize(self):
        return super(AttentionFramesBiLSTM, self).ContextEmbeddingSize + \
               self.Config.AttentionModel.AttentionEmbeddingSize

    # endregion

    def get_att_input(self):
        return self.get_input_parameter(InputSample.I_FRAME_INDS)

    # region public 'init' methods

    def init_hidden_states(self):
        super(AttentionFramesBiLSTM, self).init_hidden_states()
        with tf.variable_scope(self.__attention_scope):
            self.Config.AttentionModel.init_hidden()

    def init_input(self):
        super(AttentionFramesBiLSTM, self).init_input()
        with tf.variable_scope(self.__attention_scope):
            self.Config.AttentionModel.init_input(p_names_with_sizes=embedding.get_ns(self))

    # endregion

    def customize_rnn_output(self, rnn_outputs, s_length):

        g = sequence.select_last_relevant_in_sequence(rnn_outputs, s_length)

        with tf.variable_scope(self.__attention_scope):
            att_e, self.__att_alphas = embedding.init_mlp_attention_embedding(
                ctx_network=self,
                mlp_att=self.Config.AttentionModel,
                keys=self.get_att_input())

        return tf.concat([g, att_e], axis=-1)

    # region public 'iter' methods

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttentionFramesBiLSTM, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield u"ATT_Weights", self.__att_alphas

    def iter_hidden_parameters(self):
        for key, value in super(AttentionFramesBiLSTM, self).iter_hidden_parameters():
            yield key, value

    # endregion
