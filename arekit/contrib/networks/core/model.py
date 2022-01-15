import collections
import logging
import numpy as np

from arekit.common.model.base import BaseModel
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.labeling import LabeledCollection

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig

from arekit.contrib.networks.core.cancellation import OperationCancellation
from arekit.contrib.networks.core.ctx_inference import InferenceContext
from arekit.contrib.networks.core.ctx_predict_log import NetworkInputDependentVariables
from arekit.contrib.networks.core.feeding.bags.collection.base import BagsCollection
from arekit.contrib.networks.core.feeding.bags.collection.multi import MultiInstanceBagsCollection
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.core.feeding.batch.base import MiniBatch
from arekit.contrib.networks.core.feeding.batch.multi import MultiInstanceMiniBatch
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.networks.core.network_callback import NetworkCallback
from arekit.contrib.networks.core.nn import NeuralNetwork
from arekit.contrib.networks.core.params import NeuralNetworkModelParams

from arekit.contrib.networks.tf_helpers.nn_states import TensorflowNetworkStatesProvider
from arekit.contrib.networks.tf_helpers.session import initialize_session

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

    def __init__(self, nn_io, network, inference_ctx, bags_collection_type, config, callback=None):
        assert(isinstance(nn_io, NeuralNetworkModelIO))
        assert(isinstance(network, NeuralNetwork))
        assert(isinstance(inference_ctx, InferenceContext))
        assert(issubclass(bags_collection_type, BagsCollection))
        assert(isinstance(callback, NetworkCallback) or callback is None)
        assert(isinstance(config, DefaultNetworkConfig))

        super(BaseTensorflowModel, self).__init__(io=nn_io)

        self.__sess = None
        self.__optimiser = None
        self.__inference_ctx = inference_ctx
        self.__network = network
        self.__callback = callback
        self.__states_provider = TensorflowNetworkStatesProvider()

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

    # region private methods

    def __set_optimiser_value(self, value):
        self.__optimiser = value

    def __set_optimiser(self):
        optimiser = self.Config.Optimiser.minimize(self.__network.Cost)
        self.__set_optimiser_value(optimiser)

    def __dispose_session(self):
        """ Tensorflow session dispose method
        """
        self.__sess.close()

    # TODO. Simplify.
    def __create_batch_by_bags_group(self, bags_group):
        if issubclass(self.__bags_collection_type, SingleBagsCollection):
            return MiniBatch(bags_group)
        if issubclass(self.__bags_collection_type, MultiInstanceBagsCollection):
            return MultiInstanceMiniBatch(bags_group)

    def __create_feed_dict(self, minibatch, data_type):
        assert(isinstance(minibatch, MiniBatch))
        assert(isinstance(data_type, DataType))

        network_input = minibatch.to_network_input(provide_labels=data_type != DataType.Test)
        if self.FeedDictShow:
            MiniBatch.debug_output(network_input)

        return self.__network.create_feed_dict(network_input, data_type)

    def __get_bags_collection(self, data_type):
        return self.__inference_ctx.BagsCollections[data_type]

    def __get_labeled_samples_collection(self, data_type):
        return self.__inference_ctx.LabeledSamplesCollections[data_type]

    def __notify_initialized(self):
        if self.__callback is not None:
            self.__callback.on_initialized(self)

    def __fit_epoch(self, bags_group_it):
        assert(isinstance(bags_group_it, collections.Iterable))

        fit_total_cost = 0
        fit_total_acc = 0
        groups_count = 0

        for bags_group in bags_group_it:
            minibatch = self.__create_batch_by_bags_group(bags_group)
            feed_dict = self.__create_feed_dict(minibatch, data_type=DataType.Train)

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
            self.__states_provider.save_model(sess=self.__sess,
                                              path_tf_prefix=self.IO.get_model_target_path_tf_prefix())

        return fit_total_cost / groups_count, fit_total_acc / groups_count

    def __label_samples(self, data_type):
        """
        Provides algorithm of opinions labeling according to model results.
        """
        assert(isinstance(data_type, DataType))

        labeled_samples = self.__get_labeled_samples_collection(data_type)
        assert(isinstance(labeled_samples, LabeledCollection))

        predict_log = NetworkInputDependentVariables()
        idh_names = []
        idh_tensors = []
        for name, tensor in self.__network.iter_input_dependent_hidden_parameters():
            idh_names.append(name)
            idh_tensors.append(tensor)

        bags_collection = self.__get_bags_collection(data_type)
        bags_per_group = self.Config.BagsPerMinibatch
        bags_group_it = bags_collection.iter_by_groups(bags_per_group=bags_per_group,
                                                       text_opinion_ids_set=None)

        batches_it = self.__callback.handle_batches_iter(
            batches_iter=bags_group_it,
            total=bags_collection.get_groups_count(bags_per_group),
            prefix="Predict [{dtype}]".format(dtype=data_type))

        for bags_group in batches_it:

            minibatch = self.__create_batch_by_bags_group(bags_group)
            feed_dict = self.__create_feed_dict(minibatch=minibatch,
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

                uint_label = int(uint_labels[bag_index])

                for sample in bag:
                    labeled_samples.assign_uint_label(uint_label, sample.ID)

        return predict_log

    # endregion

    def run_training(self, model_params, seed):
        assert(isinstance(model_params, NeuralNetworkModelParams))
        self.__network.compile(self.Config, reset_graph=True, graph_seed=seed)
        self.__set_optimiser()
        self.__notify_initialized()
        self.__sess = initialize_session()

        if self.IO.IsPretrainedStateProvided:
            self.__states_provider.load_model(sess=self.__sess,
                                              path_tf_prefix=self.IO.get_model_source_path_tf_prefix())

        self.fit(epochs_count=model_params.EpochsCount)
        self.__dispose_session()

    def fit(self, epochs_count):
        assert(isinstance(epochs_count, int))
        assert(self.__sess is not None)
        assert(isinstance(self.__callback, NetworkCallback))

        operation_cancel = OperationCancellation()
        bags_collection = self.__get_bags_collection(DataType.Train)

        bags_per_group = self.Config.BagsPerMinibatch
        # However this is not a precise value.
        minibatches_count = bags_collection.get_groups_count(bags_per_group)
        logger.info("Minibatches passing per epoch count: ~{} "
                    "(Might be greater or equal, as the last "
                    "bag is expanded)".format(minibatches_count))

        if self.__callback is not None:
            # This might be used to perform
            # evaluation for original state.
            self.__callback.on_fit_started(operation_cancel)

        for epoch_index in range(epochs_count):

            if operation_cancel.IsCancelled:
                break

            bags_collection.shuffle()

            bags_group_it = self.__callback.handle_batches_iter(
                batches_iter=bags_collection.iter_by_groups(bags_per_group=bags_per_group),
                total=minibatches_count,
                prefix="Training")

            e_fit_cost, e_fit_acc = self.__fit_epoch(bags_group_it)

            if self.__callback is not None:
                self.__callback.on_epoch_finished(avg_fit_cost=e_fit_cost,
                                                  avg_fit_acc=e_fit_acc,
                                                  epoch_index=epoch_index,
                                                  operation_cancel=operation_cancel)

        if self.__callback is not None:
            self.__callback.on_fit_finished()

    def predict(self, data_type=DataType.Test, do_compile=False, graph_seed=0):
        """ Fills the related labeling collection.
        """

        # Optionally perform network compilation
        if do_compile:
            self.__network.compile(config=self.Config,
                                   reset_graph=True,
                                   graph_seed=graph_seed)

        if self.IO.IsPretrainedStateProvided:
            self.__sess = initialize_session()
            self.__states_provider.load_model(sess=self.__sess,
                                              path_tf_prefix=self.IO.get_model_source_path_tf_prefix())

        labeled_samples = self.__get_labeled_samples_collection(data_type=data_type)

        # Clear and assert the correctness.
        labeled_samples.reset_labels()
        assert(labeled_samples.is_empty())

        # Guarantee and initialize session if the latter was not.
        if self.__sess is None:
            self.__sess = initialize_session()

        return self.__label_samples(data_type=data_type)

    def get_hidden_parameters(self):
        names = []
        tensors = []
        for name, tensor in self.__network.iter_hidden_parameters():
            names.append(name)
            tensors.append(tensor)

        result_list = self.__sess.run(tensors)
        return names, result_list

    def get_labeled_samples_collection(self, data_type):
        return self.__get_labeled_samples_collection(data_type)