import tensorflow as tf

from core.networks.context.architectures.att_ends_cnn import AttentionCNN
from core.networks.context.architectures.pcnn import PiecewiseCNN
from core.networks.context.sample import InputSample


class AttentionFramesPCNN(PiecewiseCNN):

    __attention_var_scope_name = 'attention-model'

    def __init__(self):
        super(AttentionFramesPCNN, self).__init__()
        self.__att_weights = None

    # region properties

    @property
    def ContextEmbeddingSize(self):
        return super(AttentionFramesPCNN, self).ContextEmbeddingSize + \
               self.Config.AttentionModel.AttentionEmbeddingSize

    # endregion

    # region init methods

    def init_input(self):
        super(AttentionFramesPCNN, self).init_input()
        with tf.variable_scope(self.__attention_var_scope_name):
            self.Config.AttentionModel.init_input()

    def init_hidden_states(self):
        super(AttentionFramesPCNN, self).init_hidden_states()
        with tf.variable_scope(self.__attention_var_scope_name):
            self.Config.AttentionModel.init_hidden()

    def init_context_embedding(self, embedded_terms):
        g = super(AttentionFramesPCNN, self).init_context_embedding(embedded_terms)

        att_e, self.__att_weights = AttentionCNN.init_attention_embedding(
            ctx_network=self,
            att=self.Config.AttentionModel,
            keys=self.get_input_parameter(InputSample.I_FRAME_INDS))

        return tf.concat([g, att_e], axis=-1)

    # endregion

    # region iter methods

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttentionFramesPCNN, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield u"ATT_Weights", self.__att_weights

    def iter_hidden_parameters(self):
        for key, value in super(AttentionFramesPCNN, self).iter_hidden_parameters():
            yield key, value

    # endregion
