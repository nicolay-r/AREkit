import collections
import os
import logging
import numpy as np
import tensorflow as tf

from tensorflow.python.training.saver import Saver

from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.experiment.labeling import LabeledCollection
from arekit.common.model.base import BaseModel
from arekit.common.experiment.data_type import DataType
from arekit.common.utils import progress_bar_defined

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig

from arekit.contrib.networks.core.callback.base import Callback
from arekit.contrib.networks.core.cancellation import OperationCancellation
from arekit.contrib.networks.core.data_handling.data import HandledData
from arekit.contrib.networks.core.data_handling.labeling import BaseSamplesLabeling
from arekit.contrib.networks.core.data_handling.predict_log import NetworkInputDependentVariables
from arekit.contrib.networks.core.feeding.bags.collection.base import BagsCollection
from arekit.contrib.networks.core.feeding.bags.collection.multi import MultiInstanceBagsCollection
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.core.feeding.batch.base import MiniBatch
from arekit.contrib.networks.core.feeding.batch.multi import MultiInstanceMiniBatch
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.networks.core.nn import NeuralNetwork

logger = logging.getLogger(__name__)


class BaseTensorflowModel(BaseModel):
    """
    Base model class, which provides api for
        - tensorflow model compilation
        - fitting
        - feeding
        - load/save states during fitting/feeding
        and more.
    """

    SaveTensorflowModelStateOnFit = True
    FeedDictShow = False

    def __init__(self, nn_io, network, label_scaler,
                 handled_data, evaluator, bags_collection_type,
                 config, callback=None):
        assert(isinstance(nn_io, NeuralNetworkModelIO))
        assert(isinstance(network, NeuralNetwork))
        assert(isinstance(label_scaler, BaseLabelScaler))
        assert(isinstance(handled_data, HandledData))
        assert(isinstance(evaluator, BaseEvaluator) or evaluator is None)
        assert(issubclass(bags_collection_type, BagsCollection))
        assert(isinstance(callback, Callback) or callback is None)
        assert(isinstance(config, DefaultNetworkConfig))

        super(BaseTensorflowModel, self).__init__(io=nn_io)

        self.__sess = None
        self.__saver = None
        self.__optimiser = None
        self.__init_helper = handled_data
        self.__network = network
        self.__callback = callback
        self.__label_scaler = label_scaler
        self.__current_epoch_index = 0

        self.__config = config
        self.__bags_collection_type = bags_collection_type

    # region Properties

    @property
    def Config(self):
        return self.__config

    @property
    def Callback(self):
        return self.__callback

    # endregion

    # region public methods

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

    def __dispose_session(self):
        """
        Tensorflow session dispose method
        """
        self.__sess.close()

    def run_training(self, epochs_count):
        self.__network.compile(self.Config, reset_graph=True)
        self.set_optimiser()
        self.__notify_initialized()

        self.__initialize_session()

        if self.IO.IsPretrainedStateProvided:
            saved_model_path = u"{}.state".format(self.IO.get_model_source_path_tf_prefix())
            logger.info(u"Loading model: {}".format(saved_model_path))
            self.load_model(saved_model_path)

        self.fit(epochs_count=epochs_count)
        self.__dispose_session()

    # endregion

    # region Abstract

    def fit(self, epochs_count):
        assert(isinstance(epochs_count, int))
        assert(self.__sess is not None)
        assert(isinstance(self.__callback, Callback))

        operation_cancel = OperationCancellation()
        bags_collection = self.get_bags_collection(DataType.Train)

        bags_per_group = self.Config.BagsPerMinibatch
        # However this is not a precise value.
        minibatches_count = bags_collection.get_groups_count(bags_per_group)
        logger.info("Minibatches passing per epoch count: ~{} (Might be greater or equal, as the last bag is expanded)".format(minibatches_count))

        if self.__callback is not None:
            # This might be used to perform
            # evaluation for original state.
            self.__callback.on_fit_started(operation_cancel)

        for epoch_index in xrange(epochs_count):

            if operation_cancel.IsCancelled:
                break

            bags_collection.shuffle()

            e_fit_cost, e_fit_acc = self.__fit_epoch(
                minibatches_iter=bags_collection.iter_by_groups(bags_per_group=bags_per_group),
                total=minibatches_count)

            if self.__callback is not None:
                self.__callback.on_epoch_finished(avg_fit_cost=e_fit_cost,
                                                  avg_fit_acc=e_fit_acc,
                                                  epoch_index=epoch_index,
                                                  operation_cancel=operation_cancel)

            self.__current_epoch_index += 1

        if self.__callback is not None:
            self.__callback.on_fit_finished()

    def predict(self, data_type=DataType.Test):
        """ Fills the related labeling collection.
        """
        labeling_collection = self.get_samples_labeling_collection(data_type=data_type)

        labeling = BaseSamplesLabeling(data_type=data_type,
                                       samples_labeling_collection=labeling_collection)

        predict_log = labeling.predict(labeling_callback=lambda: self.__samples_labeling(data_type=data_type))

        return predict_log

    def get_hidden_parameters(self):
        names = []
        tensors = []
        for name, tensor in self.__network.iter_hidden_parameters():
            names.append(name)
            tensors.append(tensor)

        result_list = self.__sess.run(tensors)
        return names, result_list

    def set_optimiser(self):
        optimiser = self.Config.Optimiser.minimize(self.__network.Cost)
        self.set_optimiser_value(optimiser)

    def get_bags_collection(self, data_type):
        return self.__init_helper.BagsCollections[data_type]

    def get_samples_labeling_collection(self, data_type):
        return self.__init_helper.SamplesLabelingCollection[data_type]

    # TODO. Simplify.
    def create_batch_by_bags_group(self, bags_group):
        if issubclass(self.__bags_collection_type, SingleBagsCollection):
            return MiniBatch(bags_group)
        if issubclass(self.__bags_collection_type, MultiInstanceBagsCollection):
            return MultiInstanceMiniBatch(bags_group)

    def create_feed_dict(self, minibatch, data_type):
        assert(isinstance(minibatch, MiniBatch))
        assert(isinstance(data_type, DataType))

        network_input = minibatch.to_network_input(label_scaler=self.__label_scaler,
                                                   provide_labels=data_type != DataType.Test)
        if self.FeedDictShow:
            MiniBatch.debug_output(network_input)

        return self.__network.create_feed_dict(network_input, data_type)

    # endregion

    # region Private

    def __fit_epoch(self, minibatches_iter, total):
        assert(isinstance(minibatches_iter, collections.Iterable))

        fit_total_cost = 0
        fit_total_acc = 0
        groups_count = 0

        it = progress_bar_defined(iterable=minibatches_iter,
                                  unit='mbs',
                                  desc="Training e={}".format(self.__current_epoch_index),
                                  total=total)

        for bags_group in it:
            minibatch = self.create_batch_by_bags_group(bags_group)
            feed_dict = self.create_feed_dict(minibatch, data_type=DataType.Train)

            hidden_list = list(self.__network.iter_hidden_parameters())
            fetches_default = [self.__optimiser, self.__network.Cost, self.__network.Accuracy]
            fetches_hidden = [tensor for _, tensor in hidden_list]

            result = self.__sess.run(fetches_default + fetches_hidden,
                                     feed_dict=feed_dict)
            cost = result[1]

            fit_total_cost += np.mean(cost)
            fit_total_acc += result[2]
            groups_count += 1

        if BaseTensorflowModel.SaveTensorflowModelStateOnFit:
            save_fp = self.IO.get_model_target_path_tf_prefix()
            logger.info(u"Update TensorFlow model state: {}".format(save_fp))
            self.save_model(save_path=save_fp)

        return fit_total_cost / groups_count, fit_total_acc / groups_count

    def __notify_initialized(self):
        if self.__callback is not None:
            self.__callback.on_initialized(self)

    def __initialize_session(self):
        """
        Tensorflow session initialization
        """
        init_op = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(init_op)
        self.__saver = tf.train.Saver(max_to_keep=2)
        self.__sess = sess

    def __samples_labeling(self, data_type):
        """
        Provides algorithm of opinions labeling according to model results.
        """
        assert(isinstance(data_type, DataType))

        labeled_samples = self.get_samples_labeling_collection(data_type)
        assert(isinstance(labeled_samples, LabeledCollection))

        predict_log = NetworkInputDependentVariables()
        idh_names = []
        idh_tensors = []
        for name, tensor in self.__network.iter_input_dependent_hidden_parameters():
            idh_names.append(name)
            idh_tensors.append(tensor)

        bags_collection = self.get_bags_collection(data_type)
        bags_per_group = self.Config.BagsPerMinibatch
        bags_group_it = bags_collection.iter_by_groups(bags_per_group=bags_per_group,
                                                       text_opinion_ids_set=None)

        it = progress_bar_defined(
            iterable=bags_group_it,
            desc="Predict e={epoch} [{dtype}]".format(epoch=self.__current_epoch_index, dtype=data_type),
            total=bags_collection.get_groups_count(bags_per_group))

        for bags_group in it:

            minibatch = self.create_batch_by_bags_group(bags_group)
            feed_dict = self.create_feed_dict(minibatch=minibatch,
                                              data_type=data_type)

            result = self.__sess.run([self.__network.Labels] + idh_tensors, feed_dict=feed_dict)
            uint_labels = result[0]
            idh_values = result[1:]

            if len(idh_names) > 0 and len(idh_values) > 0:
                predict_log.add_input_dependent_values(names_list=idh_names,
                                                       tensor_values_list=idh_values,
                                                       text_opinion_ids=[sample.ID for sample in
                                                                         minibatch.iter_by_samples()],
                                                       bags_per_minibatch=self.Config.BagsPerMinibatch,
                                                       bag_size=self.Config.BagSize)

            # apply labeling
            for bag_index, bag in enumerate(minibatch.iter_by_bags()):

                label = self.__label_scaler.uint_to_label(value=int(uint_labels[bag_index]))

                for sample in bag:
                    if sample.ID < 0:
                        continue
                    labeled_samples.apply_label(label, sample.ID)

        return predict_log

    # endregion
