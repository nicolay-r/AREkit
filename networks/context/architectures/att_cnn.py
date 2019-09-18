import tensorflow as tf

from core.networks.attention.architectures.mlp import MultiLayerPerceptronAttention
from core.networks.context.configurations.att_cnn import AttentionCNNConfig
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

    @property
    def ContextEmbeddingSize(self):
        return super(AttentionCNN, self).ContextEmbeddingSize + \
               self.Config.AttentionModel.AttentionEmbeddingSize

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
        return tf.concat([g, self.__init_attention_embedding()], axis=-1)

    def iter_input_dependent_hidden_parameters(self):
        yield u"ATT_Weights", self.__att_weights

    def iter_hidden_parameters(self):
        for key, value in super(AttentionCNN, self).iter_hidden_parameters():
            yield key, value

    def __init_attention_embedding(self):
        assert(isinstance(self.Config, AttentionCNNConfig))

        att = self.Config.AttentionModel

        assert(isinstance(att, MultiLayerPerceptronAttention))

        # TODO. Update it with frames.
        # TODO. Create different attention.
        # TODO. att_frames_cnn.
        # TODO. att_frames_pcnn.
        entities = tf.stack([self.get_input_parameter(InputSample.I_SUBJ_IND),
                             self.get_input_parameter(InputSample.I_OBJ_IND)],
                            axis=-1)

        # TODO. Create another model att_cnn_dists.py
        # TODO. Create another model att_cnn_dists.py
        # TODO. How to: pass obj_dists
        # TODO. How to: pass subj_dists
        # TODO. How to: pass dist_emb in init_body
        att.set_input(x=self.get_input_parameter(InputSample.I_X_INDS),
                      entities=entities)

        # TODO. Here would be interesting to catch also distances.
        # TODO. The latter might give more information when we crossing frames with words.
        # TODO. Better: frames+dist with words+dists.
        # TODO. And globally to catch other context specific info.
        att_e, self.__att_weights = att.init_body(self.TermEmbedding)
        return att_e

