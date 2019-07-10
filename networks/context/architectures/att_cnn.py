import tensorflow as tf

from core.networks.attention.architectures.yatian import AttentionYatianColing2016
from core.networks.context.configurations.att_cnn import AttentionCNNConfig
from core.networks.context.sample import InputSample
from core.networks.context.architectures.cnn import VanillaCNN


class AttentionCNN(VanillaCNN):

    __attention_var_scope_name = 'attention-model'

    def __init__(self):
        super(AttentionCNN, self).__init__()
        self.__att_weights = None

    @property
    def ContextEmbeddingSize(self):
        return super(AttentionCNN, self).ContextEmbeddingSize + \
               self.Config.AttentionModel.AttentionEmbeddingSize

    def init_context_embedding_core(self, embedded_terms):
        g = super(AttentionCNN, self).init_context_embedding_core(embedded_terms)

        with tf.variable_scope(self.__attention_var_scope_name):
            self.__cfg.AttentionModel.init_hidden()

        g = tf.concat([g, self.__init_attention_embedding()], axis=-1)

        return g

    def init_input(self):
        super(AttentionCNN, self).init_input()

        with tf.variable_scope(self.__attention_var_scope_name):
            self.__cfg.AttentionModel.init_input()

    def iter_hidden_parameters(self):
        for key, value in super(AttentionCNN, self).iter_hidden_parameters():
            yield key, value

        yield "ATT_Weights", self.__att_weights

    def __init_attention_embedding(self):
        assert(isinstance(self.__cfg, AttentionCNNConfig))

        att = self.__cfg.AttentionModel

        assert(isinstance(att, AttentionYatianColing2016))

        entities = tf.stack([self.__input[InputSample.I_SUBJ_IND],
                             self.__input[InputSample.I_OBJ_IND]],
                            axis=-1)

        att.set_input(x=self.__input[InputSample.I_X_INDS],
                      entities=entities)

        att_e, self.__att_weights = att.init_body(self.__term_emb)
        return att_e

