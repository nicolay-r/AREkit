import os
import tensorflow as tf
from tensorflow.python.training.saver import Saver
from core.networks.callback import Callback
from core.networks.network_io import NetworkIO
from core.networks.network import NeuralNetwork
from core.networks.context.training.data_type import DataType


class TensorflowModel(object):
    """
    Base model class, which provides api for
        - tensorflow model compilation
        - fitting
        - training
        - load/save states during fitting/training
        and more.
    """

    def __init__(self, io, network, callback=None):
        assert(isinstance(io, NetworkIO))
        assert(isinstance(network, NeuralNetwork))
        assert(isinstance(callback, Callback) or callback is None)
        self.__sess = None
        self.__saver = None
        self.__optimiser = None
        self.__io = io
        self.__network = network
        self.__callback = callback

    # region Properties

    @property
    def Config(self):
        raise NotImplementedError()

    @property
    def Session(self):
        return self.__sess

    @property
    def Callback(self):
        return self.__callback

    @property
    def Network(self):
        return self.__network

    @property
    def Optimiser(self):
        return self.__optimiser

    @property
    def IO(self):
        return self.__io

    # endregion

    def set_optimiser_value(self, value):
        self.__optimiser = value

    def load_model(self, save_path):
        assert(isinstance(self.__saver, Saver))
        save_dir = os.path.dirname(save_path)
        self.__saver.restore(sess=self.__sess,
                             save_path=tf.train.latest_checkpoint(save_dir))

    def save_model(self, save_path):
        assert(isinstance(self.__saver, Saver))
        self.__saver.save(self.__sess,
                          save_path=save_path,
                          write_meta_graph=False)

    def dispose_session(self):
        """
        Tensorflow session dispose method
        """
        self.__sess.close()

    def run(self, load_model=False):
        self.__network.compile(self.Config, reset_graph=True)
        self.set_optimiser()
        self.__notify_initialized()

        self.__initialize_session()

        if load_model:
            save_path = self.__io.create_model_state_filepath()
            print "Loading model: {}".format(save_path)
            self.load_model(save_path)

        self.fit()
        self.dispose_session()

    # region Abstract

    def fit(self):
        raise NotImplementedError()

    def predict(self, dest_data_type=DataType.Test):
        raise NotImplementedError()

    def set_optimiser(self):
        raise NotImplementedError()

    def get_gpu_memory_fraction(self):
        raise NotImplementedError()

    # endregion

    # region Private

    def __notify_initialized(self):
        if self.__callback is not None:
            self.__callback.on_initialized(self)

    def __initialize_session(self):
        """
        Tensorflow session initialization
        """
        init_op = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.get_gpu_memory_fraction())
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(init_op)
        self.__saver = tf.train.Saver(max_to_keep=2)
        self.__sess = sess

    # endregion
