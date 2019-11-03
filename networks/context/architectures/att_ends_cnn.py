import tensorflow as tf

from core.networks.attention.architectures.cnn_attention_mlp import MultiLayerPerceptronAttention
from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.sample import InputSample
from core.networks.context.architectures.cnn import VanillaCNN


class AttentionCNN(VanillaCNN):
    """
    Author: Yatian Shen, Xuanjing Huang
    Paper: https://www.aclweb.org/anthology/C16-1238
    """

    __attention_var_scope_name = 'attention-model'

    def __init__(self):
        super(AttentionCNN, self).__init__()
        self.__att_weights = None

    # region properties

    @property
    def ContextEmbeddingSize(self):
        return super(AttentionCNN, self).ContextEmbeddingSize + \
               self.Config.AttentionModel.AttentionEmbeddingSize

    # endregion

    def set_att_weights(self, weights):
        self.__att_weights = weights

    def get_att_input(self):
        return self.get_input_entity_pairs()

    # region 'init' methods

    def init_input(self):
        super(AttentionCNN, self).init_input()
        with tf.variable_scope(self.__attention_var_scope_name):
            self.Config.AttentionModel.init_input()

    def init_hidden_states(self):
        super(AttentionCNN, self).init_hidden_states()
        with tf.variable_scope(self.__attention_var_scope_name):
            self.Config.AttentionModel.init_hidden()

    def init_context_embedding_core(self, embedded_terms):
        g = super(AttentionCNN, self).init_context_embedding_core(embedded_terms)

        att_e, att_weights = AttentionCNN.init_attention_embedding(
            ctx_network=self,
            att=self.Config.AttentionModel,
            keys=self.get_att_input())

        self.set_att_weights(att_weights)

        return tf.concat([g, att_e], axis=-1)

    # endregion

    # region iter methods

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttentionCNN, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield u"ATT_Weights", self.__att_weights

    def iter_hidden_parameters(self):
        for key, value in super(AttentionCNN, self).iter_hidden_parameters():
            yield key, value

    # endregion

    @staticmethod
    def init_attention_embedding(ctx_network, att, keys):
        assert(isinstance(ctx_network, BaseContextNeuralNetwork))
        assert(isinstance(att, MultiLayerPerceptronAttention))

        att.set_input(x=ctx_network.get_input_parameter(InputSample.I_X_INDS),
                      pos=ctx_network.get_input_parameter(InputSample.I_POS_INDS),
                      dist_obj=ctx_network.get_input_parameter(InputSample.I_OBJ_DISTS),
                      dist_subj=ctx_network.get_input_parameter(InputSample.I_SUBJ_DISTS),
                      keys=keys)

        att_e, att_w = att.init_body(
            term_embedding=ctx_network.TermEmbedding,
            pos_embedding=ctx_network.POSEmbedding,
            dist_embedding=ctx_network.DistanceEmbedding)

        return att_e, att_w

