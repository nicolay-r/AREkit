import os
import numpy as np
import tensorflow as tf
from tensorflow.python.training.saver import Saver
from core.networks.callback import Callback
from core.networks.cancellation import OperationCancellation
from core.networks.context.debug import DebugKeys
from core.networks.context.training.batch import MiniBatch
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
        assert(self.Session is not None)

        operation_cancel = OperationCancellation()
        minibatches = list(self.get_bags_collection(DataType.Train).iter_by_groups(self.Config.BagsPerMinibatch))
        print "Minibatches passing per epoch count: {}".format(len(minibatches))

        for epoch_index in range(self.Config.Epochs):

            if operation_cancel.IsCancelled:
                break

            e_cost, e_acc = self.__fit_epoch(minibatches=minibatches)

            if self.Callback is not None:
                self.Callback.on_epoch_finished(avg_cost=e_cost,
                                                avg_acc=e_acc,
                                                epoch_index=epoch_index,
                                                operation_cancel=operation_cancel)

        if self.Callback is not None:
            self.Callback.on_fit_finished()

    def predict(self, dest_data_type=DataType.Test):
        raise NotImplementedError()

    def set_optimiser(self):
        optimiser = self.Config.Optimiser.minimize(self.Network.Cost)
        self.set_optimiser_value(optimiser)

    def get_gpu_memory_fraction(self):
        raise NotImplementedError()

    def create_batch_by_bags_group(self, bags_group):
        raise NotImplementedError()

    def create_feed_dict(self, minibatch, data_type):
        assert(isinstance(self.Network, NeuralNetwork))
        assert(isinstance(minibatch, MiniBatch))
        assert(isinstance(data_type, unicode))

        input = minibatch.to_network_input()
        if DebugKeys.FeedDictShow:
            MiniBatch.debug_output(input)

        return self.Network.create_feed_dict(input, data_type)

    # endregion

    # region Private

    def __fit_epoch(self, minibatches):
        assert(isinstance(minibatches, list))

        # self.get_bags_collection_helper(DataType.Train).print_log_statistics()

        total_cost = 0
        total_acc = 0
        groups_count = 0

        np.random.shuffle(minibatches)

        for bags_group in minibatches:

            minibatch = self.create_batch_by_bags_group(bags_group)
            feed_dict = self.create_feed_dict(minibatch, data_type=DataType.Train)

            hidden_list = list(self.Network.iter_hidden_parameters())
            hidden_names = [name for name, _ in hidden_list]
            fetches_default = [self.Optimiser, self.Network.Cost, self.Network.Accuracy]
            fetches_hidden = [tensor for _, tensor in hidden_list]

            result = self.Session.run(fetches_default + fetches_hidden,
                                      feed_dict=feed_dict)
            cost = result[1]

            if DebugKeys.FitBatchDisplayLog:
                self.__display_log(hidden_names, result[len(fetches_default):])

            total_cost += np.mean(cost)
            total_acc += result[2]
            groups_count += 1

        if DebugKeys.FitSaveTensorflowModelState:
            self.save_model(save_path=self.IO.get_model_filepath())

        return total_cost / groups_count, \
               total_acc / groups_count

    @staticmethod
    def __display_log(log_names, log_values):
        assert(len(log_names) == len(log_values))
        print '==========================================='
        for index, log_value in enumerate(log_values):
            print "{}: {}".format(log_names[index], log_value)
        print '==========================================='

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
