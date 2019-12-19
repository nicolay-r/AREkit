import tensorflow as tf

from arekit.networks.attention.helpers import embedding
from arekit.networks.context.architectures.cnn import VanillaCNN


class AttentionCNNBase(VanillaCNN):
    """
    Author: Yatian Shen, Xuanjing Huang
    Paper: https://www.aclweb.org/anthology/C16-1238

    Represents a base (abstract) class with attention scope.
    Usage:
        implement `get_att_input` method in nested class.
        configuration should include AttentionModel.
    """

    __attention_scope_name = 'attention-model'
    __attention_weights_log_parameter = u"ATT_Weights"

    def __init__(self):
        super(AttentionCNNBase, self).__init__()
        self.__att_weights = None

    # region properties

    @property
    def ContextEmbeddingSize(self):
        return super(AttentionCNNBase, self).ContextEmbeddingSize + \
               self.Config.AttentionModel.AttentionEmbeddingSize

    # endregion

    def set_att_weights(self, weights):
        self.__att_weights = weights

    def get_att_input(self):
        """
        This is an abstract method which is considered to be implemented in nested class.
        """
        raise NotImplementedError()

    # region public 'init' methods

    def init_input(self):
        super(AttentionCNNBase, self).init_input()

        with tf.variable_scope(self.__attention_scope_name):
            self.Config.AttentionModel.init_input(p_names_with_sizes=embedding.get_ns(self))

    def init_hidden_states(self):
        super(AttentionCNNBase, self).init_hidden_states()
        with tf.variable_scope(self.__attention_scope_name):
            self.Config.AttentionModel.init_hidden()

    def init_context_embedding_core(self, embedded_terms):
        g = super(AttentionCNNBase, self).init_context_embedding_core(embedded_terms)

        att_e, att_weights = embedding.init_mlp_attention_embedding(
            ctx_network=self,
            mlp_att=self.Config.AttentionModel,
            keys=self.get_att_input())

        self.set_att_weights(att_weights)

        return tf.concat([g, att_e], axis=-1)

    # endregion

    # region public 'iter' methods

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttentionCNNBase, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield self.__attention_weights_log_parameter, self.__att_weights

    # endregion
