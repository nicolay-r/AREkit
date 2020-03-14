from arekit.common.opinions.collection import OpinionCollection
from arekit.contrib.experiments.experiment_io import BaseExperimentNeuralNetworkIO
from arekit.contrib.experiments.single import log
from arekit.contrib.experiments.single.evaluator import CustomOpinionBasedModelEvaluator
from arekit.contrib.experiments.single.initialization import SingleInstanceModelInitializer

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.networks.context.training.batch import MiniBatch
from arekit.networks.callback import Callback
from arekit.networks.data_type import DataType
from arekit.networks.tf_model import TensorflowModel
from arekit.networks.nn import NeuralNetwork


class SingleInstanceTensorflowModel(TensorflowModel):
    """
    This model assumes to perform a classification of a single sentence (instance, or context)
    with an attitude mentioned in it.
    """

    def __init__(self, nn_io, network, config, evaluator_class, callback):
        assert(isinstance(nn_io, BaseExperimentNeuralNetworkIO))
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(network, NeuralNetwork))
        assert(isinstance(callback, Callback) or callback is None)
        assert(callable(evaluator_class))

        super(SingleInstanceTensorflowModel, self).__init__(
            nn_io=nn_io, network=network, callback=callback)

        self.__config = config
        self.__evaluator_class = evaluator_class
        self.__init_helper = None
        self.__eval_helper = None
        self.__prepare_sources()

    # region properties

    @property
    def Config(self):
        return self.__config

    # endregion

    # region Tensorflow Model

    def get_bags_collection(self, data_type):
        return self.__init_helper.BagsCollections[data_type]

    def get_text_opinions_collection(self, data_type):
        return self.__init_helper.TextOpinionCollections[data_type]

    def get_gpu_memory_fraction(self):
        return self.__config.GPUMemoryFraction

    def get_labels_helper(self):
        return self.__init_helper.LabelsHelper

    def get_eval_helper(self):
        return self.__eval_helper

    def before_evaluation(self, dest_data_type):
        helper = self.get_text_opinions_collection_helper(dest_data_type)

        helper.debug_labels_statistic()

        collections_iter = helper.iter_opinion_collections(
            create_collection_func=lambda: self.IO.create_opinion_collection(),
            label_calculation_mode=self.Config.TextOpinionLabelCalculationMode)

        for collection, news_id in collections_iter:
            assert(isinstance(collection, OpinionCollection))

            if self.IO.EvalOnRuSentRelDocsOnly and not self.IO.is_rusentrel_news_id(news_id):
                continue

            filepath = self.IO.create_result_opinion_collection_filepath(data_type=dest_data_type,
                                                                         doc_id=news_id,
                                                                         epoch_index=self.CurrentEpochIndex)
            collection.save_to_file(filepath)

    def create_batch_by_bags_group(self, bags_group):
        return MiniBatch(bags_group)

    def create_model_init_helper(self):
        return SingleInstanceModelInitializer(nn_io=self.IO, config=self.Config)

    # endregion

    # region private methods

    def get_text_opinions_collection_helper(self, data_type):
        return self.__init_helper.TextOpinionCollectionHelpers[data_type]

    def get_bags_collection_helper(self, data_type):
        return self.__init_helper.BagsCollectionHelpers[data_type]

    def __prepare_sources(self):
        self.__init_helper = self.create_model_init_helper()

        self.__eval_helper = CustomOpinionBasedModelEvaluator(
            evaluator=self.__evaluator_class(synonyms=self.IO.DataIO.SynonymsCollection),
            nn_io=self.IO)

        self.__print_statistic()

    def __print_statistic(self):
        keys, values = self.Config.get_parameters()
        log.write_log(nn_io=self.IO, log_names=keys, log_values=values)
        self.get_text_opinions_collection_helper(DataType.Train).debug_labels_statistic()
        self.get_text_opinions_collection_helper(DataType.Train).debug_unique_relations_statistic()
        self.get_text_opinions_collection_helper(DataType.Test).debug_labels_statistic()
        self.get_text_opinions_collection_helper(DataType.Test).debug_unique_relations_statistic()
        self.get_bags_collection_helper(DataType.Train).print_log_statistics()
        self.get_bags_collection_helper(DataType.Test).print_log_statistics()

    # endregion
