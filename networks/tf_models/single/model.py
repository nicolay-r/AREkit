from arekit.common.experiment.formats.base import BaseExperiment

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
            evaluator=experiment.DataIO.Evaluator,
            callback=callback)

        self.__config = config
        self.__experiment = experiment
        self.__init_helper = self.create_model_init_helper()

        self.__print_statistic()

    # region properties

    @property
    def Config(self):
        return self.__config

    # endregion

    # region Tensorflow Model

    def get_bags_collection(self, data_type):
        return self.__init_helper.BagsCollections[data_type]

    def get_samples_labeling_collection(self, data_type):
        return self.__init_helper.SamplesLabelingCollection[data_type]

    def get_gpu_memory_fraction(self):
        return self.__config.GPUMemoryFraction

    def create_batch_by_bags_group(self, bags_group):
        return MiniBatch(bags_group)

    def create_model_init_helper(self):
        return SingleInstanceModelExperimentInitializer(experiment=self.__experiment,
                                                        config=self.Config)

    # endregion

    # region private methods

    def __print_statistic(self):
        keys, values = self.Config.get_parameters()
        log.write_log(data_io=self.__experiment.DataIO, log_names=keys, log_values=values)

        # log.debug_labels_statistic(
        #     collection=self.get_text_opinions_collection(DataType.Train),
        #     name=unicode(DataType.Train),
        #     # TODO. Labels helper assumes to be removed since all the labels presented in tsv.
        #     labels_helper=self.get_labels_helper(),
        #     stat_func=self.__init_helper.get_statistic)
        # log.debug_unique_relations_statistic(
        #     name=unicode(DataType.Train),
        #     collection=self.get_text_opinions_collection(DataType.Train))

        # log.debug_labels_statistic(
        #     collection=self.get_text_opinions_collection(DataType.Test),
        #     name=unicode(DataType.Test),
        #     # TODO. Labels helper assumes to be removed since all the labels presented in tsv.
        #     labels_helper=self.get_labels_helper(),
        #     stat_func=self.__init_helper.get_statistic)
        # log.debug_unique_relations_statistic(
        #     name=unicode(DataType.Test),
        #     collection=self.get_text_opinions_collection(DataType.Test))

    # endregion
