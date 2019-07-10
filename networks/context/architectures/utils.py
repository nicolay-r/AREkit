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


def init_accuracy(labels, true_labels):
    correct = tf.equal(labels, true_labels)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def init_weighted_cost(logits_unscaled_dropout, true_labels, config):
    """
    Init loss with weights for tensorflow model.
    'labels' suppose to be a list of indices (not priorities)
    """
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_unscaled_dropout,
        labels=true_labels)

    weights = tf.reduce_sum(
        config.ClassWeights * tf.one_hot(indices=true_labels, depth=config.ClassesCount),
        axis=1)

    if config.UseClassWeights:
        cost = cost * weights

    return weights, cost
