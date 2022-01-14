import tensorflow as tf


def initialize_session():
    """ Tensorflow session initialization
    """
    init_op = tf.compat.v1.global_variables_initializer()
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    sess.run(init_op)
    return sess
