import tensorflow as tf

from core.networks.attention.architectures.cnn_attention_mlp import MultiLayerPerceptronAttention
from core.networks.context.architectures.pcnn import PiecewiseCNN
from core.networks.context.configurations.contrib.att_pcnn import AttentionPCNNConfig
from core.networks.context.sample import InputSample


class AttentionPCNN(PiecewiseCNN):

    __attention_var_scope_name = 'attention-model'

    def __init__(self):
        super(AttentionPCNN, self).__init__()
        self.__att_weights = None

    @property
    def ContextEmbeddingSize(self):
        return super(AttentionPCNN, self).ContextEmbeddingSize + \
               self.Config.AttentionModel.AttentionEmbeddingSize

    def init_input(self):
        super(AttentionPCNN, self).init_input()
        with tf.variable_scope(self.__attention_var_scope_name):
            self.Config.AttentionModel.init_input()

    def init_hidden_states(self):
        super(AttentionPCNN, self).init_hidden_states()
        with tf.variable_scope(self.__attention_var_scope_name):
            self.Config.AttentionModel.init_hidden()

    def init_context_embedding_core(self, embedded_terms):
        g = super(AttentionPCNN, self).init_context_embedding_core(embedded_terms)
        return tf.concat([g, self.__init_attention_embedding()], axis=-1)

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(AttentionPCNN, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield u"ATT_Weights", self.__att_weights

    def iter_hidden_parameters(self):
        for key, value in super(AttentionPCNN, self).iter_hidden_parameters():
            yield key, value

    def __init_attention_embedding(self):
        assert(isinstance(self.Config, AttentionPCNNConfig))

        att = self.Config.AttentionModel

        assert(isinstance(att, MultiLayerPerceptronAttention))

        entities = tf.stack([self.get_input_parameter(InputSample.I_SUBJ_IND),
                             self.get_input_parameter(InputSample.I_OBJ_IND)],
                            axis=-1)

        att.set_input(x=self.get_input_parameter(InputSample.I_X_INDS),
                      pos=self.get_input_parameter(InputSample.I_POS_INDS),
                      dist_obj=self.get_input_parameter(InputSample.I_OBJ_DISTS),
                      dist_subj=self.get_input_parameter(InputSample.I_SUBJ_DISTS),
                      keys=entities)

        att_e, self.__att_weights = att.init_body(
            term_embedding=self.TermEmbedding,
            pos_embedding=self.POSEmbedding,
            dist_embedding=self.DistanceEmbedding)

        return att_e