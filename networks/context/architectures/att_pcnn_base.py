import tensorflow as tf

from arekit.networks.attention import common
from arekit.networks.attention.helpers import embedding
from arekit.networks.context.architectures.pcnn import PiecewiseCNN


class AttentionPCNNBase(PiecewiseCNN):
    """
    Represents a base (abstract) class with attention scope.
    Usage:
        implement `get_att_input` method in nested class.
        configuration should include AttentionModel.
    """

    def __init__(self):
        super(AttentionPCNNBase, self).__init__()
        self.__att_weights = None

    # region properties

    @property
    def ContextEmbeddingSize(self):
        return super(AttentionPCNNBase, self).ContextEmbeddingSize + \
               self.Config.AttentionModel.AttentionEmbeddingSize

    # endregion

    def set_att_weights(self, weights):
        self.__att_weights = weights

    def get_att_input(self):
        """
        This is an abstract method which is considered to be implemented in nested class.
        """
        raise NotImplementedError

    # region public `init` methods

    def init_input(self):
        super(AttentionPCNNBase, self).init_input()
        with tf.variable_scope(common.ATTENTION_SCOPE_NAME):
            self.Config.AttentionModel.init_input(p_names_with_sizes=embedding.get_ns(self))

    def init_body_dependent_hidden_states(self):
        super(AttentionPCNNBase, self).init_body_dependent_hidden_states()
        with tf.variable_scope(common.ATTENTION_SCOPE_NAME):
            self.Config.AttentionModel.init_hidden()

    def init_context_embedding(self, embedded_terms):
        g = super(AttentionPCNNBase, self).init_context_embedding(embedded_terms)

        att_e, att_weights = embedding.init_mlp_attention_embedding(
            ctx_network=self,
            mlp_att=self.Config.AttentionModel,
            keys=self.get_att_input())

        self.set_att_weights(att_weights)

        return tf.concat([g, att_e], axis=-1)

    # endregion

    # region public 'iter' methods

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttentionPCNNBase, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield common.ATTENTION_WEIGHTS_LOG_PARAMETER, self.__att_weights

    # endregion
