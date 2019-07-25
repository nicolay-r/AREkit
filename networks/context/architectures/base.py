import tensorflow as tf
from core.networks.context.architectures.utils import init_weighted_cost, init_accuracy
from core.networks.context.configurations.base import DefaultNetworkConfig
from core.networks.context.training.batch import MiniBatch
from core.networks.context.sample import InputSample
from core.networks.context.training.data_type import DataType
from core.networks.network import NeuralNetwork


# TODO. Rename SingleContextNeuralNetwork
class BaseContextNeuralNetwork(NeuralNetwork):

    def __init__(self):
        self.__cfg = None

        self.__labels = None

        self.__term_emb = None
        self.__dist_emb = None
        self.__pos_emb = None

        self.__cost = None
        self.__accuracy = None

        self.__input = {}

        self.__y = None
        self.__dropout_keep_prob = None
        self.__embedding_dropout_keep_prob = None

    # region property

    @property
    def Config(self):
        return self.__cfg

    @property
    def Labels(self):
        return self.__labels

    @property
    def Accuracy(self):
        return self.__accuracy

    @property
    def Cost(self):
        return self.__cost

    @property
    def TermEmbeddingSize(self):
        size = self.__cfg.TermEmbeddingShape[1] + 2 * self.__cfg.DistanceEmbeddingSize + 1

        if self.__cfg.UsePOSEmbedding:
            size += self.__cfg.PosEmbeddingSize

        return size

    @property
    def EmbeddingDropoutKeepProb(self):
        return self.__embedding_dropout_keep_prob

    @property
    def DropoutKeepProb(self):
        return self.__dropout_keep_prob

    @property
    def ContextEmbeddingSize(self):
        raise NotImplementedError()

    @property
    def TermEmbedding(self):
        return self.__term_emb

    # endregion

    def get_input_parameter(self, param):
        return self.__input[param]

    def set_input_parameter(self, param, value):
        self.__input[param] = value

    def set_input_dropout_keep_prob(self, value):
        self.__dropout_keep_prob = value

    def set_input_embedding_dropout_keep_prob(self, value):
        self.__embedding_dropout_keep_prob = value

    # region body

    def compile(self, config, reset_graph):
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(reset_graph, bool))

        self.__cfg = config

        if reset_graph:
            tf.reset_default_graph()

        self.init_input()
        self.__init_embedding_hidden_states()
        self.init_hidden_states()

        embedded_terms = self.init_embedded_input()
        context_embedding = self.init_context_embedding(embedded_terms)
        logits_unscaled, logits_unscaled_dropped = self.init_logits_unscaled(context_embedding)

        # Get output for each sample
        output = tf.nn.softmax(logits_unscaled)
        # Create labels only for whole bags
        self.__labels = tf.cast(tf.argmax(self.__to_mean_of_bag(output), axis=1), tf.int32)

        # TODO. to the methods.

        with tf.name_scope("cost"):
            self.__cost = init_weighted_cost(
                logits_unscaled_dropout=self.__to_mean_of_bag(logits_unscaled_dropped),
                true_labels=self.__y,
                config=config)

        # TODO. to the methods.

        with tf.name_scope("accuracy"):
            self.__accuracy = init_accuracy(labels=self.Labels, true_labels=self.__y)

    # endregion

    # region init

    def init_hidden_states(self):
        raise NotImplementedError()

    def init_context_embedding(self, embedded_terms):
        raise NotImplementedError()

    def init_logits_unscaled(self, context_embedding):
        raise NotImplementedError()

    def init_embedded_input(self):
        return self.optional_process_embedded_data(self.__cfg,
                                                   self.__init_embedded_terms(),
                                                   self.__embedding_dropout_keep_prob)

    def init_input(self):
        """
        Input placeholders
        """
        prefix = 'ctx_'

        self.__input[InputSample.I_X_INDS] = tf.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_X_INDS)

        self.__input[InputSample.I_SUBJ_DISTS] = tf.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_SUBJ_IND)

        self.__input[InputSample.I_OBJ_DISTS] = tf.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_OBJ_IND)

        self.__input[InputSample.I_TERM_TYPE] = tf.placeholder(
            dtype=tf.float32,
            shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_OBJ_IND)

        self.__input[InputSample.I_POS_INDS] = tf.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_POS_INDS)

        self.__input[InputSample.I_SUBJ_IND] = tf.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize],
            name=prefix + InputSample.I_SUBJ_IND)

        self.__input[InputSample.I_OBJ_IND] = tf.placeholder(
            dtype=tf.int32,
            shape=[self.__cfg.BatchSize],
            name=prefix + InputSample.I_OBJ_IND)

        self.__y = tf.placeholder(dtype=tf.int32,
                                  shape=[self.__cfg.BagsPerMinibatch],
                                  name=prefix + MiniBatch.I_LABELS)

        self.__dropout_keep_prob = tf.placeholder(dtype=tf.float32,
                                                  name="ctx_dropout_keep_prob")

        self.__embedding_dropout_keep_prob = tf.placeholder(dtype=tf.float32,
                                                            name="cxt_emb_dropout_keep_prob")

    # endregion

    def create_feed_dict(self, input, data_type):
        assert(isinstance(input, dict))

        feed_dict = {}
        for param in InputSample.iter_parameters():
            if param not in self.__input:
                continue
            feed_dict[self.__input[param]] = input[param]

        feed_dict[self.__y] = input[MiniBatch.I_LABELS]
        feed_dict[self.__dropout_keep_prob] = self.__cfg.DropoutKeepProb if data_type == DataType.Train else 1.0
        feed_dict[self.__embedding_dropout_keep_prob] = self.__cfg.EmbeddingDropoutKeepProb if data_type == DataType.Train else 1.0,

        return feed_dict

    # region static methods

    @staticmethod
    def optional_process_embedded_data(config, embedded, dropout_keep_prob):
        assert(isinstance(config, DefaultNetworkConfig))

        if config.UseEmbeddingDropout:
            return tf.nn.dropout(embedded, keep_prob=dropout_keep_prob)

        return embedded

    # endregion

    # region private methods

    def __to_mean_of_bag(self, logits):
        loss = tf.reshape(logits, [self.__cfg.BagsPerMinibatch, self.__cfg.BagSize, self.__cfg.ClassesCount])
        return tf.reduce_mean(loss, axis=1)

    def __init_embedded_terms(self):
        embedded_terms = tf.concat([tf.nn.embedding_lookup(self.__term_emb, self.__input[InputSample.I_X_INDS]),
                                    tf.nn.embedding_lookup(self.__dist_emb, self.__input[InputSample.I_SUBJ_DISTS]),
                                    tf.nn.embedding_lookup(self.__dist_emb, self.__input[InputSample.I_OBJ_DISTS]),
                                    tf.reshape(self.__input[InputSample.I_TERM_TYPE],
                                               [self.__cfg.BatchSize, self.__cfg.TermsPerContext, 1])],
                                   axis=-1)

        if self.__cfg.UsePOSEmbedding:
            embedded_terms = tf.concat([embedded_terms,
                                        tf.nn.embedding_lookup(self.__pos_emb, self.__input[InputSample.I_POS_INDS])],
                                       axis=-1)

        return embedded_terms

    def __init_embedding_hidden_states(self):
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

    # endregion