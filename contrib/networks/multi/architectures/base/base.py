import tensorflow as tf

from arekit.common.experiment.data_type import DataType
from arekit.contrib.networks.context.architectures.base.base import SingleInstanceNeuralNetwork
from arekit.contrib.networks.core.feeding.batch.multi import MultiInstanceMiniBatch
from arekit.contrib.networks.core.nn import NeuralNetwork
from arekit.contrib.networks.multi.configurations.base import BaseMultiInstanceConfig
from arekit.contrib.networks.sample import InputSample
from arekit.contrib.networks.tf_helpers.initialization import init_accuracy, init_weighted_cost


class BaseMultiInstanceNeuralNetwork(NeuralNetwork):

    __ctx_network_scope = "ctx_network"

    def __init__(self, context_network):
        assert(isinstance(context_network, SingleInstanceNeuralNetwork))

        self.__context_network = context_network
        self.__cfg = None

        self.__labels = None
        self.__cost = None
        self.__accuracy = None

        self.__input = {}

        self.__y = None
        self.__dropout_keep_prob = None
        self.__dropout_emb_keep_prob = None
        self.__dropout_rnn_keep_prob = None

    # region properties

    @property
    def ContextNetwork(self):
        return self.__context_network

    @property
    def Config(self):
        return self.__cfg

    @property
    def ContextsPerOpinion(self):
        return self.__cfg.BagSize

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
    def DropoutKeepProb(self):
        return self.__dropout_keep_prob

    # endregion

    # region body

    def compile(self, config, reset_graph):
        assert(isinstance(config, BaseMultiInstanceConfig))

        self.__cfg = config
        tf.reset_default_graph()

        with tf.variable_scope(self.__ctx_network_scope):
            self.__context_network.compile_hidden_states_only(config=config.ContextConfig)

        self.init_input()
        self.init_hidden_states()

        context_outputs = self.init_body()
        embedded_contexts = self.init_multiinstance_embedding(context_outputs)
        logits_unscaled, logits_unscaled_dropped = self.init_logits_unscaled(
            encoded_contexts=embedded_contexts)

        output = tf.nn.softmax(logits_unscaled)
        self.__labels = tf.cast(tf.argmax(output, axis=1), tf.int32)

        with tf.name_scope("cost"):
            self.__cost = init_weighted_cost(
                logits_unscaled_dropout=logits_unscaled_dropped,
                true_labels=self.__y,
                config=config)

        with tf.name_scope("accuracy"):
            self.__accuracy = init_accuracy(labels=self.__labels, true_labels=self.__y)

    def init_body(self):
        """
        return: [batches, sentences, embedding]
        """
        assert(isinstance(self.__cfg, BaseMultiInstanceConfig))

        with tf.name_scope("mi-body"):

            def __process_opinion(i, opinions, opinion_lens, results):
                """
                i, *, *
                """
                opinion = tf.gather(opinions, [i], axis=0)              # [1, sentences, embedding]
                opinion = tf.squeeze(opinion, [0])                      # [sentences, embedding]
                opinion_len = tf.gather(opinion_lens, [i], axis=0)      # [len]

                opinion = tf.reshape(opinion, [self.ContextsPerOpinion, self.__context_network.ContextEmbeddingSize])

                slice_size = tf.pad(opinion_len, [[0, 1]], constant_values=self.__get_context_embedding_size(opinion))
                slice_size = tf.cast(slice_size, dtype=tf.int32)

                opinion = tf.slice(opinion, [0, 0], slice_size)

                pad_len = self.ContextsPerOpinion - opinion_len         # [s-len]
                pad_len = tf.pad(pad_len, [[1, 2]])                     # [0, s-len, 0, 0]
                pad_len = tf.reshape(pad_len, [2, 2])                   # [[0, s-len] [0, 0]]
                pad_len = tf.cast(pad_len, dtype=tf.int32)

                result = tf.pad(tensor=opinion,
                                paddings=pad_len,
                                constant_values=-1)                     # [s, embedding]
                outputs = results.write(i, result)

                i += 1
                return (i, opinions, opinion_lens, outputs)

            def __process_context(i, context_embeddings):
                """
                *, i, *
                Context handler.
                """
                def __to_ctx(tensor):
                    """
                    v: [batch, contexts, embedding] -> [batch, embedding]
                    """
                    return tf.squeeze(tf.gather(tensor, [i], axis=1), [1])

                for param, value in self.__input.iteritems():
                    self.__context_network.set_input_parameter(param=param,
                                                               value=__to_ctx(value))

                self.__context_network.set_input_dropout_keep_prob(self.__dropout_keep_prob)
                self.__context_network.set_input_embedding_dropout_keep_prob(self.__dropout_emb_keep_prob)
                self.__context_network.set_input_rnn_keep_prob(self.__dropout_rnn_keep_prob)

                self.__context_network.update_network_specific_parameters()

                with tf.name_scope(self.__ctx_network_scope):
                    embedded_terms = self.__context_network.init_embedded_input()
                    context_embedding = self.__context_network.init_context_embedding(embedded_terms)

                return i + 1, context_embeddings.write(i, context_embedding)

            def __condition_process_contexts(i, context_embeddings):
                return i < self.ContextsPerOpinion

            def __iter_x_by_contexts(handler):
                context_embeddings_arr = tf.TensorArray(
                    dtype=tf.float32,
                    name="contexts_arr",
                    size=self.ContextsPerOpinion,
                    infer_shape=False,
                    dynamic_size=True)

                _, context_embeddings = tf.while_loop(
                    __condition_process_contexts,
                    handler,
                    [0, context_embeddings_arr])

                return context_embeddings.stack()

            def __iter_x_by_opinions(x, handler, opinion_lens):
                """
                x:            [batch_size, sentences, embedding]
                opinion_lens: [batch_size, len]
                """
                context_iter = tf.TensorArray(
                    dtype=tf.float32,
                    name="context_iter",
                    size=self.__cfg.BatchSize,
                    infer_shape=False,
                    dynamic_size=True)

                _, _, _, output = tf.while_loop(
                    lambda i, *_: tf.less(i, self.__cfg.BatchSize),
                    handler,
                    (0, x, opinion_lens, context_iter))

                return output.stack()

            """
            Body
            """
            context_outputs = __iter_x_by_contexts(__process_context)        # [sentences, batches, embedding]
            context_outputs = tf.transpose(context_outputs, perm=[1, 0, 2])  # [batches, sentences, embedding]
            sliced_contexts = __iter_x_by_opinions(
                x=context_outputs,
                handler=__process_opinion,
                opinion_lens=self.__calculate_opinion_lens(self.__input[InputSample.I_X_INDS]))

            return sliced_contexts

    def iter_hidden_parameters(self):
        for name, value in self.ContextNetwork.iter_hidden_parameters():
            yield u'ctx_{}'.format(name), value

    # endregion

    # region static methods

    @staticmethod
    def __calculate_opinion_lens(x):
        relevant = tf.sign(tf.abs(x))
        reduced_sentences = tf.reduce_max(relevant, reduction_indices=-1)
        length = tf.reduce_sum(reduced_sentences, reduction_indices=-1)
        length = tf.cast(length, tf.int64)
        return length

    @staticmethod
    def __get_context_embedding_size(opinion):
        return opinion.get_shape().as_list()[-1]

    # endregion

    def init_input(self):
        """
        These parameters actually are the same as in ctx model, but the shape has
        contexts count -- extra parameter.
        """
        contexts_count = self.__cfg.BagSize
        batch_size = self.__cfg.BagsPerMinibatch

        prefix = 'mi_'

        self.__input[InputSample.I_X_INDS] = tf.placeholder(
            dtype=tf.int32,
            shape=[batch_size, contexts_count, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_X_INDS)

        self.__input[InputSample.I_SUBJ_DISTS] = tf.placeholder(
            dtype=tf.int32,
            shape=[batch_size, contexts_count, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_SUBJ_DISTS)

        self.__input[InputSample.I_SYN_SUBJ_INDS] = tf.placeholder(
            dtype=tf.int32,
            shape=[batch_size, contexts_count, self.__cfg.SynonymsPerContext],
            name=prefix + InputSample.I_SYN_SUBJ_INDS)

        self.__input[InputSample.I_NEAREST_SUBJ_DISTS] = tf.placeholder(
            dtype=tf.int32,
            shape=[batch_size, contexts_count, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_NEAREST_SUBJ_DISTS)

        self.__input[InputSample.I_OBJ_DISTS] = tf.placeholder(
            dtype=tf.int32,
            shape=[batch_size, contexts_count, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_OBJ_DISTS)

        self.__input[InputSample.I_SYN_OBJ_INDS] = tf.placeholder(
            dtype=tf.int32,
            shape=[batch_size, contexts_count, self.__cfg.SynonymsPerContext],
            name=prefix + InputSample.I_SYN_OBJ_INDS)

        self.__input[InputSample.I_NEAREST_OBJ_DISTS] = tf.placeholder(
            dtype=tf.int32,
            shape=[batch_size, contexts_count, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_NEAREST_OBJ_DISTS)

        self.__input[InputSample.I_TERM_TYPE] = tf.placeholder(
            dtype=tf.float32,
            shape=[batch_size, contexts_count, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_TERM_TYPE)

        self.__input[InputSample.I_POS_INDS] = tf.placeholder(
            dtype=tf.int32,
            shape=[batch_size, contexts_count, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_POS_INDS)

        self.__input[InputSample.I_SUBJ_IND] = tf.placeholder(
            dtype=tf.int32,
            shape=[batch_size, contexts_count],
            name=prefix + InputSample.I_SUBJ_IND)

        self.__input[InputSample.I_OBJ_IND] = tf.placeholder(
            dtype=tf.int32,
            shape=[batch_size, contexts_count],
            name=prefix + InputSample.I_OBJ_IND)

        self.__input[InputSample.I_FRAME_SENT_ROLES] = tf.placeholder(
            dtype=tf.int32,
            shape=[batch_size, contexts_count, self.__cfg.TermsPerContext],
            name=prefix + InputSample.I_FRAME_SENT_ROLES
        )

        self.__input[InputSample.I_FRAME_INDS] = tf.placeholder(
            dtype=tf.int32,
            shape=[batch_size, contexts_count, self.__cfg.FramesPerContext],
            name=prefix + InputSample.I_FRAME_INDS
        )

        self.__y = tf.placeholder(dtype=tf.int32,
                                  shape=[batch_size],
                                  name=prefix + 'Y')
        self.__dropout_keep_prob = tf.placeholder(
            dtype=tf.float32,
            name=prefix + 'dropout_keep_prob')
        self.__dropout_emb_keep_prob = tf.placeholder(
            dtype=tf.float32,
            name=prefix + 'dropout_emb_keep_prob')
        self.__dropout_rnn_keep_prob = tf.placeholder(
            dtype=tf.float32,
            name=prefix + 'dropout_rnn_keep_prob')

    def iter_input_dependent_hidden_parameters(self):
        for name, value in super(BaseMultiInstanceNeuralNetwork, self).iter_input_dependent_hidden_parameters():
            yield name, value

        yield u'x', self.__input[InputSample.I_X_INDS]
        yield u'obj_ind', self.__input[InputSample.I_OBJ_IND]
        yield u'subj_ind', self.__input[InputSample.I_SUBJ_IND]
        yield u'frame_inds', self.__input[InputSample.I_FRAME_INDS]
        yield u'frame_sent_role_inds', self.__input[InputSample.I_FRAME_SENT_ROLES]
        yield u'y_labels', self.Labels
        yield u'y_etalon_labels', self.__y

    def create_feed_dict(self, input, data_type):
        assert(isinstance(input, dict))

        feed_dict = {}

        for param in InputSample.iter_parameters():
            if param not in self.__input:
                raise Exception('parameter "{}" was not found in __input. Check the presence of related parameter in "mi" initialization.'.format(param))
            feed_dict[self.__input[param]] = input[param]

        feed_dict[self.__y] = input[MultiInstanceMiniBatch.I_LABELS]
        feed_dict[self.__dropout_keep_prob] = self.__cfg.DropoutKeepProb \
            if data_type == DataType.Train else 1.0
        feed_dict[self.__dropout_emb_keep_prob] = self.__cfg.EmbeddingDropoutKeepProb \
            if data_type == DataType.Train else 1.0
        # TODO. Utilize correct dropout value in config.
        feed_dict[self.__dropout_rnn_keep_prob] = 0.9 \
            if data_type == DataType.Train else 1.0

        return feed_dict

    # region Not implemented

    def init_hidden_states(self):
        raise NotImplementedError()

    def init_logits_unscaled(self, encoded_contexts):
        raise NotImplementedError()

    def init_multiinstance_embedding(self, context_outputs):
        """
        context_outputs: [batches, sentences, embedding]
        """
        raise NotImplementedError()

    # endregion
