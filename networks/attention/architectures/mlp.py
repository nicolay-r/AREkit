import tensorflow as tf

from core.networks.attention.configurations.mlp import MultiLayerPerceptronAttentionConfig
from core.networks.context.architectures.utils import get_k_layer_logits


class MultiLayerPerceptronAttention(object):
    """
    Authors: Yatian Shen, Xuanjing Huang
    Paper: https://www.aclweb.org/anthology/C16-1238
    """

    H_W_we = "H_W_we"
    H_W_a = "H_W_a"
    H_b_we = "H_b_we"
    H_b_a = "H_b_a"

    I_x = "I_x"
    I_pos = "I_pos"
    I_dist_obj = "I_dist_obj"
    I_dist_subj = "I_dist_subj"
    I_entities = "I_e"

    def __init__(self, cfg, batch_size, terms_per_context,
                 term_embedding_size,
                 pos_embedding_size,
                 dist_embedding_size):
        assert(isinstance(cfg, MultiLayerPerceptronAttentionConfig))
        self.__cfg = cfg

        self.__batch_size = batch_size
        self.__terms_per_context = terms_per_context
        self.__term_embedding_size = term_embedding_size + \
                                     pos_embedding_size + \
                                     2 * dist_embedding_size

        self.__input = {}
        self.__hidden = {}

    @property
    def Input(self):
        return self.__input

    @property
    def AttentionEmbeddingSize(self):
        return self.__cfg.EntitiesPerContext * self.__term_embedding_size

    def set_input(self, x, pos, dist_obj, dist_subj, keys):
        self.__input[self.I_x] = x
        self.__input[self.I_pos] = pos
        self.__input[self.I_dist_subj] = dist_subj
        self.__input[self.I_dist_obj] = dist_obj
        self.__input[self.I_entities] = keys

    def init_input(self):
        self.__input[self.I_x] = tf.placeholder(dtype=tf.int32,
                                                shape=[self.__batch_size, self.__terms_per_context])
        self.__input[self.I_pos] = tf.placeholder(dtype=tf.int32,
                                                  shape=[self.__batch_size, self.__terms_per_context])
        self.__input[self.I_dist_obj] = tf.placeholder(dtype=tf.int32,
                                                       shape=[self.__batch_size, self.__terms_per_context])
        self.__input[self.I_dist_subj] = tf.placeholder(dtype=tf.int32,
                                                        shape=[self.__batch_size, self.__terms_per_context])
        self.__input[self.I_entities] = tf.placeholder(dtype=tf.int32,
                                                       shape=[self.__batch_size, self.__cfg.EntitiesPerContext])

    def init_hidden(self):
        self.__hidden[self.H_W_we] = tf.Variable(tf.random_normal([2 * self.__term_embedding_size, self.__cfg.HiddenSize]),
                                                 dtype=tf.float32)
        self.__hidden[self.H_b_we] = tf.Variable(tf.random_normal([self.__cfg.HiddenSize]),
                                                 dtype=tf.float32)
        self.__hidden[self.H_W_a] = tf.Variable(tf.random_normal([self.__cfg.HiddenSize, 1]),
                                                dtype=tf.float32)
        self.__hidden[self.H_b_a] = tf.Variable(tf.random_normal([1]),
                                                dtype=tf.float32)

    def init_body(self, term_embedding, pos_embedding, dist_embedding):
        assert(isinstance(term_embedding, tf.Tensor))

        embedded_terms = tf.concat(
            [tf.nn.embedding_lookup(params=term_embedding, ids=self.__input[self.I_x]),
             tf.nn.embedding_lookup(params=pos_embedding, ids=self.__input[self.I_pos]),
             tf.nn.embedding_lookup(params=dist_embedding, ids=self.__input[self.I_dist_subj]),
             tf.nn.embedding_lookup(params=dist_embedding, ids=self.__input[self.I_dist_obj])],
            axis=-1)

        with tf.name_scope("attention"):

            def filter_batch_elements(elements, inds, handler):
                """
                elements:  [batch_size, terms_per_context]
                """
                batch_size = elements.shape[0]

                filtered = tf.TensorArray(
                    dtype=tf.int32,
                    name="context_iter",
                    size=batch_size,
                    infer_shape=False,
                    dynamic_size=True)

                _, _, _, filtered = tf.while_loop(
                    lambda i, *_: tf.less(i, batch_size),
                    handler,
                    (0, elements, inds, filtered))

                return filtered.stack()

            def select_entity_related_elements(i, elements, inds, filtered):
                """
                elements: [batch, terms_per_context]
                inds: [batch, terms_per_context]
                """

                row_elements = tf.squeeze(tf.gather(elements, [i], axis=0))
                row_inds = tf.squeeze(tf.gather(inds, [i], axis=0))

                result = tf.gather(row_elements, row_inds)   # row: [entities_per_context]

                return (i + 1,
                        elements,
                        inds,
                        filtered.write(i, tf.squeeze(result)))

            def iter_by_entities(entities, e_pos, e_dist_obj, e_dist_subj, handler):
                """
                entities:  [batch_size, entities]
                e_pos:     [batch_size, terms_per_context]
                e_dists:   [batch_size, terms_per_context]
                """

                att_sum_array = tf.TensorArray(
                    dtype=tf.float32,
                    name="context_iter",
                    size=self.__cfg.EntitiesPerContext,
                    infer_shape=False,
                    dynamic_size=True)

                att_weights_array = tf.TensorArray(
                    dtype=tf.float32,
                    name="context_iter",
                    size=self.__cfg.EntitiesPerContext,
                    infer_shape=False,
                    dynamic_size=True)

                _, _, _, _, _, att_sum, att_weights = tf.while_loop(
                    lambda i, *_: tf.less(i, self.__cfg.EntitiesPerContext),
                    handler,
                    (0, entities, e_pos, e_dist_obj, e_dist_subj, att_sum_array, att_weights_array))

                return att_sum.stack(), att_weights.stack()

            def process_entity(i, entities, e_pos, e_dist_obj, e_dist_subj, att_sum, att_weights):
                """
                entities: [batch_size, entities_per_context]
                """

                e_term_index = tf.gather(entities, [i], axis=1)                         # [batch_size, 1] -- term positions
                e_term_indices = tf.tile(e_term_index, [1, self.__terms_per_context])   # [batch_size, terms_per_context]

                e_pos_indices = filter_batch_elements(
                    elements=e_pos,
                    inds=e_term_indices,
                    handler=select_entity_related_elements)

                e_dist_obj_indices = filter_batch_elements(
                    elements=e_dist_obj,
                    inds=e_term_indices,
                    handler=select_entity_related_elements)

                e_dist_subj_indices = filter_batch_elements(
                    elements=e_dist_subj,
                    inds=e_term_indices,
                    handler=select_entity_related_elements)

                e = tf.concat(
                    [tf.nn.embedding_lookup(term_embedding, e_term_indices),            # [batch_size, terms_per_context, embedding_size]
                     tf.nn.embedding_lookup(pos_embedding, e_pos_indices),
                     tf.nn.embedding_lookup(dist_embedding, e_dist_obj_indices),
                     tf.nn.embedding_lookup(dist_embedding, e_dist_subj_indices)],
                    axis=-1)                                                            # [batch_size, terms_per_context, embedding_size]

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
                        entities, e_pos, e_dist_obj, e_dist_subj,
                        att_sum.write(i, w_sum),
                        att_weights.write(i, tf.reshape(alphas, [self.__batch_size, self.__terms_per_context])))

            att_sum, att_weights = iter_by_entities(
                entities=self.__input[self.I_entities],
                e_pos=self.__input[self.I_pos],
                e_dist_obj=self.__input[self.I_dist_obj],
                e_dist_subj=self.__input[self.I_dist_subj],
                handler=process_entity)

            # att_sum: [entity_per_context, batch_size, term_embedding_size]
            # att_weights: [entity_per_context, batch_size, terms_per_context]

            att_sum = tf.transpose(att_sum, perm=[1, 0, 2])  # [batch_size, entity_per_context, term_embedding_size]
            att_sum = tf.reshape(att_sum, shape=[self.__batch_size, self.AttentionEmbeddingSize])

            att_weights = tf.transpose(att_weights, perm=[1, 0, 2])  # [batch_size, entity_per_context, terms_per_context]

        return att_sum, att_weights
