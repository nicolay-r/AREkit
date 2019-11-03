import tensorflow as tf

from core.networks.context.architectures.att_ends_cnn import AttentionCNN
from core.networks.context.architectures.pcnn import PiecewiseCNN


class AttentionAttitudeEndsPCNN(PiecewiseCNN):

    __attention_var_scope_name = 'attention-model'

    def __init__(self):
        super(AttentionAttitudeEndsPCNN, self).__init__()
        self.__att_weights = None

    # region properties

    @property
    def ContextEmbeddingSize(self):
        return super(AttentionAttitudeEndsPCNN, self).ContextEmbeddingSize + \
               self.Config.AttentionModel.AttentionEmbeddingSize

    # endregion

    def set_att_weights(self, weights):
        self.__att_weights = weights

    def get_att_input(self):
        return self.get_input_entity_pairs()

    # region init methods

    def init_input(self):
        super(AttentionAttitudeEndsPCNN, self).init_input()
        with tf.variable_scope(self.__attention_var_scope_name):
            self.Config.AttentionModel.init_input()

    def init_hidden_states(self):
        super(AttentionAttitudeEndsPCNN, self).init_hidden_states()
        with tf.variable_scope(self.__attention_var_scope_name):
            self.Config.AttentionModel.init_hidden()

    def init_context_embedding(self, embedded_terms):
        g = super(AttentionAttitudeEndsPCNN, self).init_context_embedding(embedded_terms)

        att_e, att_weights = AttentionCNN.init_attention_embedding(
            ctx_network=self,
            att=self.Config.AttentionModel,
            keys=self.get_att_input())

        self.set_att_weights(att_weights)

        return tf.concat([g, att_e], axis=-1)

    # endregion

    # region iter methods

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttentionAttitudeEndsPCNN, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield u"ATT_Weights", self.__att_weights

    def iter_hidden_parameters(self):
        for key, value in super(AttentionAttitudeEndsPCNN, self).iter_hidden_parameters():
            yield key, value

    # endregion