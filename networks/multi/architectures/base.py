import tensorflow as tf

from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.architectures.utils import get_two_layer_logits
from core.networks.context.training.sample import Sample
from core.networks.context.training.data_type import DataType
from core.networks.multi.configuration.base import MIMLRESettings
from core.networks.multi.training.batch import MultiInstanceBatch
from core.networks.network import NeuralNetwork


class MIMLRE(NeuralNetwork):

    _context_network_scope_name = "context_network"

    def __init__(self, context_network):
        assert(isinstance(context_network, BaseContextNeuralNetwork))
        self.context_network = context_network
        self.cfg = None

        self.__W1 = None
        self.__W2 = None
        self.__b1 = None
        self.__b2 = None

        self.__labels = None
        self.__weights = None
        self.__cost = None
        self.__accuracy = None

        # TODO. Use dictionary for input parameters.
        self.__x = None
        self.__y = None
        self.__dist_from_subj = None
        self.__dist_from_obj = None
        self.__term_type = None
        self.__pos = None
        self.__p_subj_ind = None
        self.__p_obj_ind = None
        self.__dropout_keep_prob = None
        self.__embedding_dropout_keep_prob = None

    # region properties

    @property
    def ContextsPerOpinion(self):
        return self.cfg.BagSize

    @property
    def Labels(self):
        return self.__labels

    @property
    def Accuracy(self):
        return self.__accuracy

    @property
    def Cost(self):
        return self.__cost

    # endregion

    # region body

    def compile(self, config, reset_graph):
        assert(isinstance(config, MIMLRESettings))

        self.cfg = config
        tf.reset_default_graph()

        with tf.variable_scope("ctx-network"):
            self.context_network.compile(config=config.ContextSettings, reset_graph=False)

        self.init_input()
        self.init_hidden_states()

        max_pooling = self.init_body()
        logits_unscaled, logits_unscaled_dropped = get_two_layer_logits(
            max_pooling,
            self.__W1, self.__b1,
            self.__W2, self.__b2,
            self.__dropout_keep_prob,
            activations=[tf.tanh, tf.tanh, None])
        output = tf.nn.softmax(logits_unscaled)
        self.__labels = tf.cast(tf.argmax(output, axis=1), tf.int32)

        with tf.name_scope("cost"):
            self.__weights, self.__cost = BaseContextNeuralNetwork.init_weighted_cost(
                logits_unscaled_dropout=logits_unscaled_dropped,
                true_labels=self.__y,
                config=config)

        with tf.name_scope("accuracy"):
            self.__accuracy = BaseContextNeuralNetwork.init_accuracy(labels=self.__labels, true_labels=self.__y)

    def init_body(self):
        assert(isinstance(self.cfg, MIMLRESettings))

        with tf.name_scope("mi-body"):

            def __process_opinion(i, opinions, opinion_lens, results):
                """
                i, *, *
                """
                opinion = tf.gather(opinions, [i], axis=0)              # [1, sentences, embedding]
                opinion = tf.squeeze(opinion, [0])                      # [sentences, embedding]
                opinion_len = tf.gather(opinion_lens, [i], axis=0)      # [len]

                opinion = tf.reshape(opinion, [self.ContextsPerOpinion, self.context_network.ContextEmbeddingSize])

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

                # TODO. Store keys in sample!
                # TODO. store variables in dictionary.
                self.context_network.set_input_x(__to_ctx(self.__x))
                self.context_network.set_input_dist_from_subj(__to_ctx(self.__dist_from_subj))
                self.context_network.set_input_dist_from_obj(__to_ctx(self.__dist_from_obj))
                self.context_network.set_input_term_type(__to_ctx(self.__term_type))
                self.context_network.set_input_pos(__to_ctx(self.__pos))
                self.context_network.set_input_p_subj_ind(__to_ctx(self.__p_subj_ind))
                self.context_network.set_input_p_obj_ind(__to_ctx(self.__p_obj_ind))
                self.context_network.set_input_dropout_keep_prob(self.__dropout_keep_prob)
                self.context_network.set_input_embedding_dropout_keep_prob(self.__embedding_dropout_keep_prob)

                embedded_terms = self.context_network.init_embedded_input()
                context_embedding = self.context_network.init_context_embedding(embedded_terms)

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
                    size=self.cfg.BatchSize,
                    infer_shape=False,
                    dynamic_size=True)

                _, _, _, output = tf.while_loop(
                    lambda i, *_: tf.less(i, self.cfg.BatchSize),
                    handler,
                    (0, x, opinion_lens, context_iter))

                return output.stack()

            """
            Body
            """
            context_outputs = __iter_x_by_contexts(__process_context)                    # [sentences, batches, embedding]
            context_outputs = tf.transpose(context_outputs, perm=[1, 0, 2])              # [batches, sentences, embedding]
            sliced_contexts = __iter_x_by_opinions(
                x=context_outputs,
                handler=__process_opinion,
                opinion_lens=self.__calculate_opinion_lens(self.__x))

            return self.__contexts_max_pooling(context_outputs=sliced_contexts,
                                               contexts_per_opinion=self.ContextsPerOpinion)  # [batches, max_pooling]

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
    def __contexts_max_pooling(context_outputs, contexts_per_opinion):
        context_outputs = tf.expand_dims(context_outputs, 0)     # [1, batches, sentences, embedding]
        context_outputs = tf.nn.max_pool(
            context_outputs,
            ksize=[1, 1, contexts_per_opinion, 1],
            strides=[1, 1, contexts_per_opinion, 1],
            padding='VALID',
            data_format="NHWC")
        return tf.squeeze(context_outputs)                       # [batches, max_pooling]

    @staticmethod
    def __get_context_embedding_size(opinion):
        return opinion.get_shape().as_list()[-1]

    # endregion

    def init_input(self):
        """
        These parameters actually are the same as in ctx model, but the shape has
        contexts count -- extra parameter.
        """
        contexts_count = self.cfg.BagSize
        batch_size = self.cfg.BagsPerMinibatch

        self.__x = tf.placeholder(dtype=tf.int32, shape=[batch_size, contexts_count, self.cfg.TermsPerContext])
        self.__y = tf.placeholder(dtype=tf.int32, shape=[batch_size])

        self.__dist_from_subj = tf.placeholder(dtype=tf.int32, shape=[batch_size, contexts_count, self.cfg.TermsPerContext])
        self.__dist_from_obj = tf.placeholder(dtype=tf.int32, shape=[batch_size, contexts_count, self.cfg.TermsPerContext])
        self.__term_type = tf.placeholder(tf.float32, shape=[batch_size, contexts_count, self.cfg.TermsPerContext])
        self.__pos = tf.placeholder(tf.int32, shape=[batch_size, contexts_count, self.cfg.TermsPerContext])
        self.__p_subj_ind = tf.placeholder(dtype=tf.int32, shape=[batch_size, contexts_count])
        self.__p_obj_ind = tf.placeholder(dtype=tf.int32, shape=[batch_size, contexts_count])

        self.__dropout_keep_prob = tf.placeholder(tf.float32)
        self.__embedding_dropout_keep_prob = tf.placeholder(tf.float32)

    def init_hidden_states(self):
        self.__W1 = tf.Variable(tf.random_normal([self.context_network.ContextEmbeddingSize, self.cfg.HiddenSize]), dtype=tf.float32)
        self.__W2 = tf.Variable(tf.random_normal([self.cfg.HiddenSize, self.cfg.ClassesCount]), dtype=tf.float32)
        self.__b1 = tf.Variable(tf.random_normal([self.cfg.HiddenSize]), dtype=tf.float32)
        self.__b2 = tf.Variable(tf.random_normal([self.cfg.ClassesCount]), dtype=tf.float32)

    def create_feed_dict(self, input, data_type):

        feed_dict = {
            self.__x: input[Sample.I_X_INDS],
            self.__y: input[MultiInstanceBatch.I_LABELS],
            self.__dist_from_subj: input[Sample.I_SUBJ_DISTS],
            self.__dist_from_obj: input[Sample.I_OBJ_DISTS],
            self.__term_type: input[Sample.I_TERM_TYPE],
            self.__p_subj_ind: input[Sample.I_SUBJ_IND],
            self.__p_obj_ind: input[Sample.I_OBJ_IND],
            self.__dropout_keep_prob: self.cfg.DropoutKeepProb if data_type == DataType.Train else 1.0,
            self.__embedding_dropout_keep_prob: self.cfg.EmbeddingDropoutKeepProb if data_type == DataType.Train else 1.0
        }

        if self.cfg.UsePOSEmbedding:
            feed_dict[self.__pos] = input[Sample.I_POS_INDS]

        return feed_dict
