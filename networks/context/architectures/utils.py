import tensorflow as tf


def calculate_sequence_length(seq):
    relevant = tf.sign(tf.abs(seq))
    length = tf.reduce_sum(relevant, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def get_k_layer_logits(g, W, b, dropout_keep_prob=None, activations=None):
    assert(isinstance(W, list))
    assert(isinstance(b, list))
    assert(isinstance(activations, list))
    assert(len(W) == len(b) == len(activations) - 1)

    def __activate(tensor, activation):
        return activation(tensor) if activation is not None else tensor

    if activations is None:
        activations = [None] * (len(W) + 1)

    r = g

    for i in range(len(W)):
        r = __activate(r, activations[0])
        r = tf.matmul(r, W[i]) + b[i]

        if dropout_keep_prob is None:
            continue

        r = tf.nn.dropout(r, dropout_keep_prob)

    return __activate(r, activations[-1])


def get_k_layer_pair_logits(g, W, b, dropout_keep_prob, activations):
    assert(dropout_keep_prob is not None)

    result = get_k_layer_logits(g, W, b, activations=activations)
    result_dropout = get_k_layer_logits(g, W, b,
                                        dropout_keep_prob=dropout_keep_prob,
                                        activations=activations)

    return result, result_dropout
