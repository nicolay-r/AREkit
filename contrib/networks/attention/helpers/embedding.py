from arekit.contrib.networks.attention.architectures.mlp import MLPAttention
from arekit.networks.context.architectures.base import SingleInstanceNeuralNetwork
from arekit.networks.context.sample import InputSample


def __get_NEVS_list(ctx_network):
    """
    Helper for additional embeddings.
    Here, NEVS stands for: name, embedding, value, size list
    """
    assert(isinstance(ctx_network, SingleInstanceNeuralNetwork))

    def ctx_network_input_or_none(p_name):
        if ctx_network.has_input_parameter(p_name):
            return ctx_network.get_input_parameter(p_name)

    i_x = "I_x"
    i_pos = "I_pos"
    i_dist_obj = "I_dist_obj"
    i_dist_subj = "I_dist_subj"
    i_nearest_dist_obj = "I_nearest_dist_obj"
    i_nearest_dist_subj = "I_nearest_dist_subj"
    i_sent_role_frames = "I_sent_role_frames"

    return [(i_x,
             ctx_network.TermEmbedding,
             ctx_network_input_or_none(InputSample.I_X_INDS),
             ctx_network.Config.TermEmbeddingShape[1]),

            (i_pos,
             ctx_network.POSEmbedding,
             ctx_network_input_or_none(InputSample.I_POS_INDS),
             ctx_network.Config.PosEmbeddingSize),

            (i_dist_obj,
             ctx_network.DistanceEmbedding,
             ctx_network_input_or_none(InputSample.I_OBJ_DISTS),
             ctx_network.Config.DistanceEmbeddingSize),

            (i_dist_subj,
             ctx_network.DistanceEmbedding,
             ctx_network_input_or_none(InputSample.I_SUBJ_DISTS),
             ctx_network.Config.DistanceEmbeddingSize),

            (i_nearest_dist_obj,
             ctx_network.DistanceEmbedding,
             ctx_network_input_or_none(InputSample.I_NEAREST_OBJ_DISTS),
             ctx_network.Config.DistanceEmbeddingSize),

            (i_nearest_dist_subj,
             ctx_network.DistanceEmbedding,
             ctx_network_input_or_none(InputSample.I_NEAREST_SUBJ_DISTS),
             ctx_network.Config.DistanceEmbeddingSize),

            (i_sent_role_frames,
             ctx_network.SentimentEmbedding,
             ctx_network_input_or_none(InputSample.I_FRAME_SENT_ROLES),
             ctx_network.Config.SentimentEmbeddingSize)
            ]


def get_nv(ctx_network):
    return [(n, v) for n, _, v, _ in __get_NEVS_list(ctx_network)]


def get_ev(ctx_network):
    return [(e, v) for _, e, v, _ in __get_NEVS_list(ctx_network)]


def get_ne(ctx_network):
    return [(n, e) for n, e, _, _ in __get_NEVS_list(ctx_network)]


def get_ns(ctx_network):
    return [(n, s) for n, _, _, s in __get_NEVS_list(ctx_network)]


def init_mlp_attention_embedding(ctx_network, mlp_att, keys):
    assert(isinstance(ctx_network, SingleInstanceNeuralNetwork))
    assert(isinstance(mlp_att, MLPAttention))

    mlp_att.set_input(param_names_with_values=get_nv(ctx_network),
                      keys=keys)

    att_e, att_w = mlp_att.init_body(params_embeddings=get_ne(ctx_network))

    return att_e, att_w

