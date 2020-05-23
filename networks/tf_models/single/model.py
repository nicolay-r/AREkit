from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.model.evaluator import CustomOpinionBasedModelEvaluator
from arekit.common.experiment.data_type import DataType

from arekit.networks.tf_models.single import log
from arekit.networks.callback import Callback
from arekit.networks.tf_models.single.initialization import SingleInstanceModelExperimentInitializer
from arekit.networks.tf_models.model import TensorflowModel
from arekit.networks.nn import NeuralNetwork
from arekit.networks.training.batch.batch import MiniBatch

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig


class SingleInstanceTensorflowModel(TensorflowModel):
    """
    This model assumes to perform a classification of a single sentence (instance, or context)
    with an attitude mentioned in it.
    """

    def __init__(self, experiment, network, config, callback):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(network, NeuralNetwork))
        assert(isinstance(callback, Callback) or callback is None)

        super(SingleInstanceTensorflowModel, self).__init__(
            nn_io=experiment.DataIO.ModelIO,
            network=network,
            label_scaler=experiment.DataIO.LabelsScaler,
            callback=callback)

        self.__config = config
        self.__experiment = experiment
        self.__init_helper = self.create_model_init_helper()

        # TODO. To base model.
        self.__evaluator = CustomOpinionBasedModelEvaluator(
            evaluator=experiment.DataIO.Evaluator,
            model=self)

        self.__print_statistic()

    # region properties

    @property
    def Config(self):
        return self.__config

    # endregion

    # region Tensorflow Model

    def get_bags_collection(self, data_type):
        return self.__init_helper.BagsCollections[data_type]

    def get_labeling_collection(self, data_type):
        return self.__init_helper.LabeledCollection[data_type]

    def get_gpu_memory_fraction(self):
        return self.__config.GPUMemoryFraction

    def get_labels_helper(self):
        return self.__init_helper.LabelsHelper

    def get_evaluator(self):
        return self.__evaluator

    def create_batch_by_bags_group(self, bags_group):
        return MiniBatch(bags_group)

    def create_model_init_helper(self):
        return SingleInstanceModelExperimentInitializer(experiment=self.__experiment,
                                                        config=self.Config)

    # endregion

    # region private methods

    def get_text_opinions_collection_helper(self, data_type):
        return self.__init_helper.TextOpinionCollectionHelpers[data_type]

    def get_bags_collection_helper(self, data_type):
        return self.__init_helper.BagsCollectionHelpers[data_type]

    def __print_statistic(self):
        keys, values = self.Config.get_parameters()
        log.write_log(data_io=self.__experiment.DataIO, log_names=keys, log_values=values)
        self.get_text_opinions_collection_helper(DataType.Train).debug_labels_statistic()
        self.get_text_opinions_collection_helper(DataType.Train).debug_unique_relations_statistic()
        self.get_text_opinions_collection_helper(DataType.Test).debug_labels_statistic()
        self.get_text_opinions_collection_helper(DataType.Test).debug_unique_relations_statistic()
        self.get_bags_collection_helper(DataType.Train).print_log_statistics()
        self.get_bags_collection_helper(DataType.Test).print_log_statistics()

    # endregion
