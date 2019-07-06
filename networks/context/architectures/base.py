import tensorflow as tf
from core.networks.context.architectures.attention.base import Attention
from core.networks.context.configurations.base import CommonModelSettings
from core.networks.context.training.batch import MiniBatch
from core.networks.context.training.sample import Sample
from core.networks.context.training.data_type import DataType
from core.networks.network import NeuralNetwork


class BaseContextNeuralNetwork(NeuralNetwork):

    __attention_var_scope_name = 'attention-model'

    def __init__(self):
        self.__cfg = None

        self.__att_weights = None
        self.__labels = None

        self.__term_emb = None
        self.__dist_emb = None
        self.__pos_emb = None

        self.__cost = None
        self.__weights = None
        self.__accuracy = None

        self.__x = None
        self.__y = None
        self.__dist_from_subj = None
        self.__dist_from_obj = None
        self.__term_type = None
        self.__pos = None
        self.__p_subj_ind = None
        self.__p_obj_ind = None
        self.dropout_keep_prob = None
        self.embedding_dropout_keep_prob = None

    @property
    def Config(self):
        return self.__cfg

    @property
    def InputX(self):
        return self.__x

    @property
    def InputPObjInd(self):
        return self.__p_obj_ind

    @property
    def InputPSubjInd(self):
        return self.__p_subj_ind

    def set_input_x(self, x):
        self.__x = x

    def set_input_y(self, y):
        self.__y = y

    def set_input_dist_from_subj(self, dist_from_subj):
        self.__dist_from_subj = dist_from_subj

    def set_input_dist_from_obj(self, dist_from_obj):
        self.__dist_from_obj = dist_from_obj

    def set_input_term_type(self, term_type):
        self.__term_type = term_type

    def set_input_pos(self, pos):
        self.__pos = pos

    def set_input_p_subj_ind(self, p_subj_ind):
        self.__p_subj_ind = p_subj_ind

    def set_input_p_obj_ind(self, p_obj_ind):
        self.__p_obj_ind = p_obj_ind

    def set_input_dropout_keep_prob(self, value):
        self.dropout_keep_prob = value

    def set_input_embedding_dropout_keep_prob(self, value):
        self.embedding_dropout_keep_prob = value

    def compile(self, config, reset_graph):
        assert(isinstance(config, CommonModelSettings))
        assert(isinstance(reset_graph, bool))

        self.__cfg = config

        if reset_graph:
            tf.reset_default_graph()

        if self.__cfg.UseAttention:
            with tf.variable_scope(self.__attention_var_scope_name):
                self.__cfg.AttentionModel.init_hidden()

        self.init_input()
        self.init_embedding_hidden_states()
        self.init_hidden_states()

        embedded_terms = self.init_embedded_input()
        context_embedding = self.init_context_embedding(embedded_terms)
        logits_unscaled, logits_unscaled_dropped = self.init_logits_unscaled(context_embedding)

        # Get output for each sample
        output = tf.nn.softmax(logits_unscaled)
        # Create labels only for whole bags
        self.__labels = tf.cast(tf.argmax(self.to_mean_of_bag(output), axis=1), tf.int32)

        with tf.name_scope("cost"):
            self.__weights, self.__cost = self.init_weighted_cost(
                logits_unscaled_dropout=self.to_mean_of_bag(logits_unscaled_dropped),
                true_labels=self.__y,
                config=config)

        with tf.name_scope("accuracy"):
            self.__accuracy = self.init_accuracy(labels=self.Labels, true_labels=self.__y)

    @property
    def ContextEmbeddingSize(self):
        raise Exception("Not implemented")

    def init_embedding_hidden_states(self):
        self.__term_emb = tf.constant(value=self.__cfg.TermEmbeddingMatrix,
                                      dtype=tf.float32,
                                      shape=self.__cfg.TermEmbeddingShape)

        self.__dist_emb = tf.get_variable(dtype=tf.float32,
                                          initializer=tf.random_normal_initializer,
                                          shape=[self.__cfg.TermsPerContext, self.__cfg.DistanceEmbeddingSize],
                                          trainable=True,
                                          name="dist_emb")

        self.__pos_emb = tf.get_variable(dtype=tf.float32,
                                         initializer=tf.random_normal_initializer,
                                         shape=[len(self.__cfg.PosTagger.pos_names), self.__cfg.PosEmbeddingSize],
                                         trainable=True,
                                         name="pos_emb")

    def init_hidden_states(self):
        raise Exception("Not implemented")

    def init_context_embedding(self, embedded_terms):
        raise Exception("Not implemented")

    def init_logits_unscaled(self, context_embedding):
        raise Exception("Not implemented")

    def init_attention_embedding(self):
        assert(isinstance(self.__cfg.AttentionModel, Attention))
        self.__cfg.AttentionModel.set_x(self.__x)
        self.__cfg.AttentionModel.set_entities(tf.stack([self.__p_subj_ind, self.__p_obj_ind], axis=-1))         # [batch_size, 2]
        att_e, self.__att_weights = self.__cfg.AttentionModel.init_body(self.__term_emb)
        return att_e

    def init_embedded_input(self):
        return self.optional_process_embedded_data(self.__cfg,
                                                   self.init_embedded_terms(),
                                                   self.embedding_dropout_keep_prob)

    @staticmethod
    def optional_process_embedded_data(config, embedded, dropout_keep_prob):
        assert(isinstance(config, CommonModelSettings))

        if config.UseEmbeddingDropout:
            return tf.nn.dropout(embedded, keep_prob=dropout_keep_prob)

        return embedded

    def init_input(self):
        """
        Input placeholders
        """
        self.__x = tf.placeholder(dtype=tf.int32, shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext], name="ctx_x")
        self.__y = tf.placeholder(dtype=tf.int32, shape=[self.__cfg.BagsPerMinibatch], name="ctx_y")
        self.__dist_from_subj = tf.placeholder(dtype=tf.int32, shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext], name="ctx_dist_from_subj")
        self.__dist_from_obj = tf.placeholder(dtype=tf.int32, shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext], name="ctx_dist_from_obj")
        self.__term_type = tf.placeholder(tf.float32, shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext], name="ctx_term_type")
        self.__pos = tf.placeholder(tf.int32, shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext], name="ctx_pos")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="ctx_dropout_keep_prob")
        self.embedding_dropout_keep_prob = tf.placeholder(tf.float32, name="cxt_emb_dropout_keep_prob")
        self.__p_subj_ind = tf.placeholder(dtype=tf.int32, shape=[self.__cfg.BatchSize], name="ctx_p_subj_ind")
        self.__p_obj_ind = tf.placeholder(dtype=tf.int32, shape=[self.__cfg.BatchSize], name="ctx_p_obj_ind")

        if self.__cfg.UseAttention:
            with tf.variable_scope(self.__attention_var_scope_name):
                self.__cfg.AttentionModel.init_input()

    @staticmethod
    def _get_attention_vector_size(cfg):
        return 0 if not cfg.UseAttention else cfg.AttentionModel.AttentionEmbeddingSize

    @property
    def TermEmbeddingSize(self):
        size = self.__cfg.TermEmbeddingShape[1] + 2 * self.__cfg.DistanceEmbeddingSize + 1

        if self.__cfg.UsePOSEmbedding:
            size += self.__cfg.PosEmbeddingSize

        return size

    def init_embedded_terms(self):

        embedded_terms = tf.concat([tf.nn.embedding_lookup(self.__term_emb, self.__x),
                                    tf.nn.embedding_lookup(self.__dist_emb, self.__dist_from_subj),
                                    tf.nn.embedding_lookup(self.__dist_emb, self.__dist_from_obj),
                                    tf.reshape(self.__term_type, [self.__cfg.BatchSize, self.__cfg.TermsPerContext, 1])],
                                   axis=-1)

        if self.__cfg.UsePOSEmbedding:
            embedded_terms = tf.concat([embedded_terms,
                                        tf.nn.embedding_lookup(self.__pos_emb, self.__pos)],
                                       axis=-1)

        return embedded_terms

    @staticmethod
    def init_accuracy(labels, true_labels):
        correct = tf.equal(labels, true_labels)
        return tf.reduce_mean(tf.cast(correct, tf.float32))

    @staticmethod
    def init_weighted_cost(logits_unscaled_dropout, true_labels, config):
        """
        Init loss with weights for tensorflow model.
        'labels' suppose to be a list of indices (not priorities)
        """
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_unscaled_dropout,
            labels=true_labels)

        weights = tf.reduce_sum(
            config.ClassWeights * tf.one_hot(indices=true_labels, depth=config.ClassesCount),
            axis=1)

        if config.UseClassWeights:
            cost = cost * weights

        return weights, cost

    def to_mean_of_bag(self, logits):
        loss = tf.reshape(logits, [self.__cfg.BagsPerMinibatch, self.__cfg.BagSize, self.__cfg.ClassesCount])
        return tf.reduce_mean(loss, axis=1)

    def create_feed_dict(self, input, data_type):

        feed_dict = {
            self.__x: input[Sample.I_X_INDS],
            self.__y: input[MiniBatch.I_LABELS],
            self.__dist_from_subj: input[Sample.I_SUBJ_DISTS],
            self.__dist_from_obj: input[Sample.I_OBJ_DISTS],
            self.__term_type: input[Sample.I_TERM_TYPE],
            self.dropout_keep_prob: self.__cfg.DropoutKeepProb if data_type == DataType.Train else 1.0,
            self.embedding_dropout_keep_prob: self.__cfg.EmbeddingDropoutKeepProb if data_type == DataType.Train else 1.0,
            self.__p_subj_ind: input[Sample.I_SUBJ_IND],
            self.__p_obj_ind: input[Sample.I_OBJ_IND]
        }

        if self.__cfg.UsePOSEmbedding:
            feed_dict[self.__pos] = input[Sample.I_POS_INDS]

        return feed_dict

    @property
    def Labels(self):
        return self.__labels

    @property
    def Accuracy(self):
        return self.__accuracy

    @property
    def Cost(self):
        return self.__cost