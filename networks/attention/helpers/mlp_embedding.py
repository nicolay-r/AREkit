from core.networks.attention.architectures.cnn_attention_mlp import MLPAttention
from core.networks.context.architectures.base import BaseContextNeuralNetwork
from core.networks.context.sample import InputSample


def __get_NEVS_list(ctx_network):
    """
    Helper for additional embeddings.
    Here, NEVS stands for: name, embedding, value, size list
    """
    assert(isinstance(ctx_network, BaseContextNeuralNetwork))

    I_x = "I_x"
    I_pos = "I_pos"
    I_dist_obj = "I_dist_obj"
    I_dist_subj = "I_dist_subj"
    # TODO. provide
    # I_sent_role_frames = "I_sent_role_frames"

    return [(I_x,
             ctx_network.TermEmbedding,
             ctx_network.get_input_parameter(InputSample.I_X_INDS),
             ctx_network.Config.TermEmbeddingShape[1]),

            (I_pos,
             ctx_network.POSEmbedding,
             ctx_network.get_input_parameter(InputSample.I_POS_INDS),
             ctx_network.Config.PosEmbeddingSize),

            (I_dist_obj,
             ctx_network.DistanceEmbedding,
             ctx_network.get_input_parameter(InputSample.I_OBJ_DISTS),
             ctx_network.Config.DistanceEmbeddingSize),

            (I_dist_subj,
             ctx_network.DistanceEmbedding,
             ctx_network.get_input_parameter(InputSample.I_SUBJ_DISTS),
             ctx_network.Config.DistanceEmbeddingSize)
            ]


def get_nv(ctx_network):
    return [(n, v) for n, _, v, _ in __get_NEVS_list(ctx_network)]


def get_ne(ctx_network):
    return [(n, e) for n, e, _, _ in __get_NEVS_list(ctx_network)]


def get_ns(ctx_network):
    return [(n, s) for n, _, _, s in __get_NEVS_list(ctx_network)]


def init_attention_embedding(ctx_network, att, keys):
    assert(isinstance(ctx_network, BaseContextNeuralNetwork))
    assert(isinstance(att, MLPAttention))

    att.set_input(param_names_with_values=get_nv(ctx_network),
                  keys=keys)

    att_e, att_w = att.init_body(params_embeddings=get_ne(ctx_network))

    return att_e, att_w

