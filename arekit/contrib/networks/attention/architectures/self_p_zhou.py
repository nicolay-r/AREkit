import tensorflow as tf


def self_attention_by_peng_zhou(inputs):
    """
    Attention method proposed by:
    Title: Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
    Authors: Peng Zhou, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao, Bo Xu
    Paper: https://www.aclweb.org/anthology/P16-2034
    Code author: SeoSangwoo (c), https://github.com/SeoSangwoo
    Code: https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction

    inputs: Tensor
        tensor of shape
    """
    assert(isinstance(inputs, tf.Tensor))

    # Trainable parameters
    hidden_size = inputs.shape[2].value
    u_omega = tf.compat.v1.get_variable(name="u_omega",
                                        shape=[hidden_size],
                                        initializer=tf.keras.initializers.glorot_normal())

    with tf.name_scope('v'):
        v = tf.tanh(inputs)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    # Final output with tanh
    output = tf.tanh(output)

    return output, alphas
