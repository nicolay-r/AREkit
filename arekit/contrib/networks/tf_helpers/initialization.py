import tensorflow as tf


def init_weighted_cost(logits_unscaled_dropout, true_labels, config):
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_unscaled_dropout,
        labels=true_labels)

    cost += tf.compat.v1.losses.get_regularization_loss()

    weights = tf.reduce_sum(
        config.ClassWeights * tf.one_hot(indices=true_labels, depth=config.ClassesCount),
        axis=1)

    if config.UseClassWeights:
        cost = cost * weights

    return cost


def init_accuracy(labels, true_labels):
    correct = tf.equal(labels, true_labels)
    return tf.reduce_mean(tf.cast(correct, tf.float32))