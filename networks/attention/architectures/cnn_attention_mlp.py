import tensorflow as tf

from core.networks.attention.configurations.cnn_attention_mlp import MultiLayerPerceptronAttentionConfig
from core.networks.tf_helpers.layers import get_k_layer_logits
from core.networks.tf_helpers.filtering import \
    filter_batch_elements, \
    select_entity_related_elements


class MLPAttention(object):
    """
    Title: Attention-Based Convolutional Neural Network for Semantic Relation Extraction
    Authors: Yatian Shen, Xuanjing Huang
    Paper: https://www.aclweb.org/anthology/C16-1238
    """

    H_W_we = "H_W_we"
    H_W_a = "H_W_a"
    H_b_we = "H_b_we"
    H_b_a = "H_b_a"

    I_keys = "I_keys"

    def __init__(self, cfg, batch_size, terms_per_context):
        assert(isinstance(cfg, MultiLayerPerceptronAttentionConfig))
        self.__cfg = cfg

        self.__batch_size = batch_size
        self.__terms_per_context = terms_per_context
        self.__term_embedding_size = None

        self.__input = {}
        self.__hidden = {}

    # region properties

    @property
    def Config(self):
        return self.__cfg

    @property
    def BatchSize(self):
        return self.__batch_size

    @property
    def TermsPerContext(self):
        return self.__terms_per_context

    @property
    def TermEmbeddingSize(self):
        return self.__term_embedding_size

    @property
    def Input(self):
        return self.__input

    @property
    def AttentionEmbeddingSize(self):
        return self.__cfg.EntitiesPerContext * self.__term_embedding_size

    # endregion

    # region public methods

    def set_input(self, param_names_with_values, keys):
        """
        param_names_with_values: list
            list of pairs <name, input>
        """
        assert(isinstance(param_names_with_values, list))

        self.__input[self.I_keys] = keys

        for p_name, value in param_names_with_values:
            self.__input[p_name] = value

    def init_input(self, p_names_with_sizes):
        """
        p_names_with_sizes: list
            list of pairs <name, size>
        """
        assert(isinstance(p_names_with_sizes, list))

        self.__term_embedding_size = sum([size for _, size in p_names_with_sizes])

        for p_name, _ in p_names_with_sizes:
            self.__input[p_name] = tf.placeholder(dtype=tf.int32,
                                                  shape=[self.__batch_size, self.__terms_per_context])

        self.__input[self.I_keys] = tf.placeholder(dtype=tf.int32,
                                                   shape=[self.__batch_size, self.__cfg.EntitiesPerContext])

    def init_hidden(self):

        self.__hidden[self.H_W_we] = tf.get_variable(
            name=self.H_W_we,
            shape=[2 * self.__term_embedding_size, self.__cfg.HiddenSize],
            initializer=self.__cfg.LayerInitializer,
            dtype=tf.float32)

        self.__hidden[self.H_b_we] = tf.get_variable(
            name=self.H_b_we,
            shape=[self.__cfg.HiddenSize],
            initializer=self.__cfg.LayerInitializer,
            dtype=tf.float32)

        self.__hidden[self.H_W_a] = tf.get_variable(
            name=self.H_W_a,
            shape=[self.__cfg.HiddenSize, 1],
            initializer=self.__cfg.LayerInitializer,
            dtype=tf.float32)

        self.__hidden[self.H_b_a] = tf.get_variable(
            name=self.H_b_a,
            shape=[1],
            initializer=self.__cfg.LayerInitializer,
            dtype=tf.float32)

    def init_body(self, params_embeddings):
        """
        params_embedding: list
            list of pairs <name, embedding>
        """
        assert(isinstance(params_embeddings, list))

        embedded_terms = tf.concat(
            values=[tf.nn.embedding_lookup(params=p_emb, ids=self.__input[p_name])
                    for p_name, p_emb in params_embeddings],
            axis=-1)

        # Parameters([(self.__input[p_name], p_emb) for p_name, p_emb in params_embeddings])

        with tf.name_scope("attention"):

            def iter_by_entities(entities, handler):
                """
                entities:  [batch_size, entities_per_context]
                handler: func
                """

                e_len = self.calculate_entities_length_func(entities)

                att_sum_array = tf.TensorArray(
                    dtype=tf.float32,
                    name="context_iter",
                    size=e_len,
                    infer_shape=False,
                    dynamic_size=True)

                att_weights_array = tf.TensorArray(
                    dtype=tf.float32,
                    name="context_iter",
                    size=e_len,
                    infer_shape=False,
                    dynamic_size=True)

                _, _, att_sum, att_weights = tf.while_loop(
                    lambda i, *_: tf.less(i, e_len),
                    handler,
                    (0, entities, att_sum_array, att_weights_array))

                return att_sum.stack(), att_weights.stack()

            def process_entity(i, entities, att_sum, att_weights):
                """
                entities: [batch_size, entities_per_context]
                params_with_embedding: list
                    list of pairs <input, embedding>
                """
                e_term_index = tf.gather(entities, [i], axis=1)                         # [batch_size, 1] -- term positions
                e_term_indices = tf.tile(e_term_index, [1, self.__terms_per_context])   # [batch_size, terms_per_context]

                embedded_params = []
                for param_name, param_embedding in params_embeddings:
                    ids = filter_batch_elements(elements=self.__input[param_name],
                                                inds=e_term_indices,
                                                handler=select_entity_related_elements)
                    embedded_params.append(tf.nn.embedding_lookup(params=param_embedding, ids=ids))

                e = tf.concat(embedded_params, axis=-1)  # [batch_size, terms_per_context, embedding_size]

                merged = tf.concat([embedded_terms, e], axis=-1)
                merged = tf.reshape(merged, [self.__batch_size * self.__terms_per_context,
                                             2 * self.__term_embedding_size])

                u = get_k_layer_logits(g=merged,
                                       W=[self.__hidden[self.H_W_we], self.__hidden[self.H_W_a]],
                                       b=[self.__hidden[self.H_b_we], self.__hidden[self.H_b_a]],
                                       activations=[None,
                                                    lambda tensor: tf.tanh(tensor),
                                                    None])       # [batch_size * terms_per_context, 1]

                alphas = tf.reshape(u, [self.__batch_size, self.__terms_per_context])
                alphas = tf.nn.softmax(alphas)
                alphas = tf.reshape(alphas, [self.__batch_size * self.__terms_per_context, 1])

                original_embedding = tf.reshape(embedded_terms,
                                                [self.__batch_size * self.__terms_per_context, self.__term_embedding_size])

                w_embedding = tf.multiply(alphas, original_embedding)
                w_embedding = tf.reshape(w_embedding, [self.__batch_size, self.__terms_per_context, self.__term_embedding_size])
                w_sum = tf.reduce_sum(w_embedding, axis=1)             # [batch_size, embedding_size]

                return (i + 1,
                        entities,
                        att_sum.write(i, w_sum),
                        att_weights.write(i, tf.reshape(alphas, [self.__batch_size, self.__terms_per_context])))

            att_sum, att_weights = iter_by_entities(
                entities=self.__input[self.I_keys],
                handler=process_entity)

            # att_sum: [entity_per_context, batch_size, term_embedding_size]
            # att_weights: [entity_per_context, batch_size, terms_per_context]
            att_weights = tf.transpose(att_weights, perm=[1, 0, 2])  # [batch_size, entity_per_context, terms_per_context]
            att_sum = tf.transpose(att_sum, perm=[1, 0, 2])  # [batch_size, entity_per_context, term_embedding_size]

        return self.reshape_att_sum(att_sum), \
               self.reshape_att_weights(att_weights)

    def reshape_att_sum(self, att_sum):
        """
        att_sum: [batch_size, entity_per_context, term_embedding_size]
        """
        att_sum = tf.reshape(att_sum, shape=[self.__batch_size, self.AttentionEmbeddingSize])
        return att_sum

    def reshape_att_weights(self, att_weights):
        """
        att_sum: [batch_size, entity_per_context, term_embedding_size]
        """
        return att_weights

    def calculate_entities_length_func(self, entities):
        """
        In this case we consider that the length is fixed to EntitiesPerContextParameter
        """
        return self.__cfg.EntitiesPerContext

    # endregion


class Parameters:

    def __init__(self, args):
        assert(isinstance(args, list))
        self.__args = args

    def iterate_pairs(self):
        for x, y in self.__args:
            yield x, y
