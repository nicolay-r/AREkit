import logging
import os
import tensorflow as tf

logger = logging.getLogger(__name__)


class TensorflowNetworkStatesProvider(object):
    """ Allows to perform tensorflow states reading and writing.
    """

    def __init__(self):
        self.__saver = None

    # region private methods

    def __init_saver(self):
        """ Note: it should be initialized once sesion has been created
            therefore we defer this action, rather than perform so at __init__
        """
        self.__saver = tf.compat.v1.train.Saver(max_to_keep=2)

    def __load_model_core(self, session, save_path):
        save_dir = os.path.dirname(save_path)

        if self.__saver is None:
            self.__init_saver()

        self.__saver.restore(sess=session, save_path=tf.train.latest_checkpoint(save_dir))

    def __save_model_core(self, session, save_path):
        if self.__saver is None:
            self.__init_saver()
        self.__saver.save(sess=session, save_path=save_path, write_meta_graph=False)

    # endregion

    def load_model(self, sess, path_tf_prefix):
        assert(isinstance(sess, tf.compat.v1.Session))

        saved_model_dir = "{}/".format(path_tf_prefix)

        if not os.path.exists(saved_model_dir):
            # Skip the case when model is not available.
            logger.info('Model was not found at: "{path}"'.format(path=saved_model_dir))
            logger.info('Skipping loading process!"')
            return

        logger.info("Loading Tensorflow model state: {}".format(saved_model_dir))
        self.__load_model_core(session=sess, save_path=saved_model_dir)

    def save_model(self, sess, path_tf_prefix):
        assert(isinstance(sess, tf.compat.v1.Session))
        logger.info("Update TensorFlow model state: {}".format(path_tf_prefix))
        self.__save_model_core(session=sess, save_path=path_tf_prefix)
