import tensorflow as tf

from arekit.contrib.networks.attention import common
from arekit.contrib.networks.context.configurations.self_att_bilstm import SelfAttentionBiLSTMConfig
from arekit.contrib.networks.sample import InputSample
from arekit.networks.data_type import DataType
from arekit.networks.tf_helpers import sequence
from arekit.contrib.networks.context.architectures.base.base import SingleInstanceNeuralNetwork


class SelfAttentionBiLSTM(SingleInstanceNeuralNetwork):
    """
    A Structured Self-attentive Sentence Embedding (ICLR 2017)
    Paper: https://arxiv.org/pdf/1703.03130.pdf
    Code Author: roomylee, https://github.com/roomylee (C)
    Code: https://github.com/roomylee/self-attentive-emb-tf
    """

    def __init__(self):
        super(SelfAttentionBiLSTM, self).__init__()

        # hidden
        self.__A = None
        self.__avg_by_r_A = None
        self.__W_s1 = None
        self.__W_s2 = None
        self.__W_output = None
        self.__b_output = None
        self.__dropout_rnn_keep_prob = None

    # region properties

    @property
    def ContextEmbeddingSize(self):
        """
        Returns: flattened M
            return r * 2u, where u is an output of a single direction in bilstm.
        """
        return self.Config.RSize * 2 * self.Config.HiddenSize

    # endregion

    # region public 'set' methods

    def set_input_rnn_keep_prob(self, value):
        self.__dropout_rnn_keep_prob = value

    # endregion

    # region public 'init' methods

    def init_input(self):
        super(SelfAttentionBiLSTM, self).init_input()
        self.__dropout_rnn_keep_prob = tf.placeholder(dtype=tf.float32,
                                                      name="ctx_dropout_rnn_keep_prob")

    def init_context_embedding(self, embedded_terms):
        assert(isinstance(self.Config, SelfAttentionBiLSTMConfig))

        # Bidirectional(Left&Right) Recurrent Structure
        with tf.name_scope("bi-lstm"):
            x_length = sequence.calculate_sequence_length(self.get_input_parameter(InputSample.I_X_INDS))
            s_length = tf.cast(x=tf.maximum(x_length, 1), dtype=tf.int32)

            fw_cell = sequence.get_cell(hidden_size=self.Config.HiddenSize,
                                        cell_type=self.Config.CellType,
                                        dropout_rnn_keep_prob=self.__dropout_rnn_keep_prob)

            bw_cell = sequence.get_cell(hidden_size=self.Config.HiddenSize,
                                        cell_type=self.Config.CellType,
                                        dropout_rnn_keep_prob=self.__dropout_rnn_keep_prob)

            (self.output_fw, self.output_bw), states = sequence.bidirectional_rnn(cell_fw=fw_cell,
                                                                                  cell_bw=bw_cell,
                                                                                  inputs=embedded_terms,
                                                                                  sequence_length=s_length,
                                                                                  dtype=tf.float32)
            H = tf.concat([self.output_fw, self.output_bw], axis=2)
            H_reshape = tf.reshape(H, [-1, 2 * self.Config.HiddenSize])

        with tf.name_scope("self-attention"):
            _H_s1 = tf.nn.tanh(tf.matmul(H_reshape, self.__W_s1))
            _H_s2 = tf.matmul(_H_s1, self.__W_s2)
            _H_s2_reshape = tf.transpose(tf.reshape(_H_s2, [-1, self.Config.TermsPerContext, self.Config.RSize]),
                                         perm=[0, 2, 1])

            self.__A = tf.nn.softmax(_H_s2_reshape, name="attention")
            self.__avg_by_r_A = tf.reduce_mean(self.__A, axis=-2)

        with tf.name_scope("sentence-embedding"):
            # M shape (r, 2u)
            M = tf.matmul(self.__A, H)

        # M_flat (batch_size, r * 2u)
        return tf.reshape(M, shape=[-1, self.ContextEmbeddingSize])

    def init_body_dependent_hidden_states(self):
        assert(isinstance(self.Config, SelfAttentionBiLSTMConfig))

        self.__W_s1 = tf.get_variable(
            name="W_s1",
            shape=[2 * self.Config.HiddenSize, self.Config.DASize],
            regularizer=self.Config.LayerRegularizer,
            initializer=self.Config.WeightInitializer)

        self.__W_s2 = tf.get_variable(
            name="W_s2",
            shape=[self.Config.DASize, self.Config.RSize],
            regularizer=self.Config.LayerRegularizer,
            initializer=self.Config.WeightInitializer)

    def init_logits_hidden_states(self):
        assert(isinstance(self.Config, SelfAttentionBiLSTMConfig))

        self.__W_output = tf.get_variable(
            name="W_output",
            shape=[self.Config.FullyConnectionSize, self.Config.ClassesCount],
            regularizer=self.Config.LayerRegularizer,
            initializer=self.Config.WeightInitializer)

        self.__b_output = tf.get_variable(
            name="b_output",
            shape=[self.Config.ClassesCount],
            regularizer=self.Config.LayerRegularizer,
            initializer=self.Config.BiasInitializer)

        self.__W_fc = tf.get_variable(
            name="W_fc",
            shape=[2 * self.Config.HiddenSize * self.Config.RSize, self.Config.FullyConnectionSize],
            regularizer=self.Config.LayerRegularizer,
            initializer=self.Config.WeightInitializer)

        self.__b_fc = tf.get_variable(
            name="b_fc",
            shape=[self.Config.FullyConnectionSize],
            regularizer=self.Config.LayerRegularizer,
            initializer=self.Config.BiasInitializer)

    def init_logits_unscaled(self, context_embedding):
        """
        context_embedding: M_flat parameter of init_context_embedding
            M_flat shape (r * 2u)
        """

        with tf.name_scope("fully-connected"):

            fc = tf.nn.relu(tf.nn.xw_plus_b(context_embedding, self.__W_fc, self.__b_fc), name="fc")

        with tf.name_scope("output"):
            logits = tf.nn.xw_plus_b(x=fc, weights=self.__W_output, biases=self.__b_output, name="logits")

        return logits, tf.nn.dropout(logits, self.DropoutKeepProb)

    def init_cost(self, logits_unscaled_dropped):
        loss = super(SelfAttentionBiLSTM, self).init_cost(logits_unscaled_dropped)

        with tf.name_scope("penalization"):
            AA_T = tf.matmul(self.__A, tf.transpose(self.__A, [0, 2, 1]))
            I = tf.reshape(tensor=tf.tile(tf.eye(self.Config.RSize), [tf.shape(self.__A)[0], 1]),
                           shape=[-1, self.Config.RSize, self.Config.RSize])
            P = tf.square(tf.norm(AA_T - I, axis=[-2, -1], ord="fro"))

        return loss + tf.reduce_mean(P * self.Config.PenaltizationTermCoef)

    # endregion

    # region public 'create' methods

    def create_feed_dict(self, input, data_type):
        feed_dict = super(SelfAttentionBiLSTM, self).create_feed_dict(input=input, data_type=data_type)
        feed_dict[self.__dropout_rnn_keep_prob] = self.Config.DropoutRNNKeepProb if data_type == DataType.Train else 1.0
        return feed_dict

    # endregion

    # region public 'iter' methods

    def iter_input_dependent_hidden_parameters(self):

        for name, value in super(SelfAttentionBiLSTM, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield common.ATTENTION_WEIGHTS_LOG_PARAMETER, self.__avg_by_r_A

    def iter_hidden_parameters(self):

        if self.__W_s1 is not None:
            yield ("W_s1", self.__W_s1)

        if self.__W_s2 is not None:
            yield ("W_s2", self.__W_s2)

        if self.__W_output is not None:
            yield ("W_output", self.__W_output)

        if self.__b_output is not None:
            yield ("b_output", self.__b_output)

    # endregion
