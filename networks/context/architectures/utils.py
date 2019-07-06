import tensorflow as tf


def calculate_sequence_length(seq):
    relevant = tf.sign(tf.abs(seq))
    length = tf.reduce_sum(relevant, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


# TODO. This shoud depends on lists of W and b params.
def get_two_layer_logits(g, W1, b1, W2, b2, dropout_keep_prob, activations=None):

    if activations is None:
        activations = [None, None, None]

    g = __optional_activate(g, activations[0])

    r1 = tf.matmul(g, W1) + b1
    r1d = tf.nn.dropout(r1, dropout_keep_prob)

    r1 = __optional_activate(r1, activations[1])
    r1d = __optional_activate(r1d, activations[1])

    r2 = tf.matmul(r1, W2) + b2
    r2d = tf.matmul(r1d, W2) + b2
    r2d = tf.nn.dropout(r2d, dropout_keep_prob)

    r2 = __optional_activate(r2, activations[2])
    r2d = __optional_activate(r2d, activations[2])

    return r2, r2d


# TODO. This should be removed as we already have method above
def get_single_layer_logits(g, W1, b1, dropout_keep_prob, activations=None):

    if activations is None:
        activations = [None, None]

    g = __optional_activate(g, activations[0])
    r = tf.matmul(g, W1) + b1
    rd = tf.nn.dropout(r, dropout_keep_prob)

    r = __optional_activate(r, activations[1])
    rd = __optional_activate(rd, activations[1])

    return r, rd

def __optional_activate(tensor, activation):
    return activation(tensor) if activation is not None else tensor
