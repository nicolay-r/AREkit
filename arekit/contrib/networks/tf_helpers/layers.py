import logging
import tensorflow as tf


logger = logging.getLogger(__name__)


def get_k_layer_logits(g, W, b, dropout_keep_prob=None, activations=None):
    assert(isinstance(W, list))
    assert(isinstance(b, list))
    assert(isinstance(activations, list) or activations is None)

    if activations is None:
        activations = [None] * (len(W) + 1)

    assert(len(W) == len(b) == len(activations) - 1)

    def __activate(tensor, activation):
        return activation(tensor) if activation is not None else tensor

    res = g

    for i in range(len(W)):

        logger.debug("LOG: r_shape={}".format(res.shape))
        logger.debug("LOG: i={}".format(i))
        logger.debug("LOG: W[i]_shape={}".format(W[i].shape))
        logger.debug("LOG: b[i]_shape={}".format(b[i].shape))

        res = __activate(res, activations[i])
        res = tf.matmul(res, W[i]) + b[i]

        if dropout_keep_prob is None:
            continue

        res = tf.nn.dropout(res, dropout_keep_prob)

    return __activate(res, activations[-1])


def get_k_layer_pair_logits(g, W, b, dropout_keep_prob, activations=None):
    assert(dropout_keep_prob is not None)

    result = get_k_layer_logits(g, W, b, activations=activations)
    result_dropout = get_k_layer_logits(g, W, b,
                                        dropout_keep_prob=dropout_keep_prob,
                                        activations=activations)

    return result, result_dropout
