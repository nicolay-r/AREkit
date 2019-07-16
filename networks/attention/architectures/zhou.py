import tensorflow as tf


class AttentionZhouACL2016:
    """
    Authors: SeoSangwoo
    Paper: https://www.aclweb.org/anthology/P16-2034
    Repository: https://github.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction
    """

    def __init__(self):
        self.__inputs = None

    def set_input(self, inputs):
        self.__inputs = inputs

    def init_body(self):
        # Trainable parameters
        hidden_size = self.__inputs.shape[2].value
        u_omega = tf.get_variable(name="u_omega",
                                  shape=[hidden_size],
                                  initializer=tf.keras.initializers.glorot_normal())

        with tf.name_scope('v'):
            v = tf.tanh(self.__inputs)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(self.__inputs * tf.expand_dims(alphas, -1), 1)

        # Final output with tanh
        output = tf.tanh(output)

        return output, alphas