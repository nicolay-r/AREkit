import collections
import logging

from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.labeling import LabeledCollection
from arekit.common.model.labeling.single import SingleLabelsHelper
from arekit.common.news.parsed.collection import ParsedNewsCollection
from arekit.common.text_opinions.text_opinion import TextOpinion

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.sample import InputSample

from arekit.networks.input.encoder import NetworkInputEncoder
from arekit.networks.input.formatters.sample import NetworkSampleFormatter
from arekit.networks.input.rows_parser import ParsedSampleRow
from arekit.networks.input.terms_mapping import EmbeddedTermMapping
from arekit.networks.training.bags.collection.single import SingleBagsCollection

logger = logging.getLogger(__name__)


class SingleInstanceModelExperimentInitializer(object):

    def __init__(self, experiment, config):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(config, DefaultNetworkConfig))

        supported_data_types = list(experiment.DocumentOperations.iter_suppoted_data_types())

        self.__string_entity_formatter = experiment.DataIO.StringEntityFormatter

        self.__pncs = self.__create_collection(
            data_types=supported_data_types,
            collection_by_dtype_func=lambda data_type: experiment.create_parsed_collection(data_type))

        self.__synonyms = experiment.DataIO.SynonymsCollection

        # TODO. Remove
        self.__labels_scaler = experiment.DataIO.LabelsScaler
        self.__labels_helper = SingleLabelsHelper(label_scaler=self.__labels_scaler)

        # TODO: Samples labeling collection (update)
        # TODO: Samples labeling collection (update)
        # TODO: Samples labeling collection (update)
        self.__labeled_collections = self.__create_collection(
            supported_data_types,
            # TODO. Labeled collection will be simplified
            lambda data_type: LabeledCollection(collection=None))

        self.__frames_collection = experiment.DataIO.FramesCollection

        # TODO. We assume here to iterate over tsv records.
        self.__bags_collection = self.__create_collection(
            supported_data_types,
            # TODO. We assume here to iterate over tsv records.
            lambda data_type: self.create_bags_collection(
                formatted_samples=None,
                config=config))

        norm, _ = self.get_statistic(collection=None,
                                     labels_helper=self.__labels_helper)

        config.set_class_weights(norm)

        config.notify_initialization_completed()

    @classmethod
    def init_from_experiment(cls, config, experiment):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(config, DefaultNetworkConfig))
        NetworkInputEncoder.to_tsv_with_embedding_and_vocabulary(config=config,
                                                                 experiment=experiment)

    # region Properties

    @property
    def _LabelsScaler(self):
        return self.__labels_scaler

    @property
    def BagsCollections(self):
        return self.__bags_collection

    @property
    def LabeledCollection(self):
        # TODO. Samples labeling collection
        return self.__labeled_collections

    # TODO. Remove
    @property
    def LabelsHelper(self):
        return self.__labels_helper

    @property
    def _BagCollectionType(self):
        return SingleBagsCollection

    # endregion

    @staticmethod
    def get_statistic(collection, labels_helper):
        stat = [0] * labels_helper.get_classes_count()

        for text_opinion in collection:
            assert(isinstance(text_opinion, TextOpinion))
            stat[labels_helper.label_to_uint(text_opinion.Sentiment)] += 1

        total = sum(stat)
        norm = [100.0 * value / total if total > 0 else 0 for value in stat]
        return norm, stat

    def _create_empty_sample_func(self, config):
        return None

    def create_bags_collection(self, formatted_samples, config):
        assert(isinstance(formatted_samples, NetworkSampleFormatter))
        assert(isinstance(config, DefaultNetworkConfig))

        collection = self._BagCollectionType.from_formatted_samples(
            formatted_samples=formatted_samples,
            bag_size=config.BagSize,
            shuffle=True,
            create_empty_sample_func=self._create_empty_sample_func,
            create_sample_func=lambda row: self.__create_input_sample(row=row, config=config))

        return collection

    # region private methods

    def __create_input_sample(self, row, config):
        """
        Creates an input for Neural Network model
        """
        assert(isinstance(row, ParsedSampleRow))
        assert(isinstance(config, DefaultNetworkConfig))

        # TODO. Provide extra parameters.
        return InputSample.from_tsv_row(
            row_id=row.RowID,
            terms=row.Terms,
            subj_ind=row.SubjectIndex,
            obj_ind=row.ObjectIndex,
            words_vocab=None,
            config=config)

    @staticmethod
    def __create_collection(data_types, collection_by_dtype_func):
        assert(isinstance(data_types, collections.Iterable))
        assert(callable(collection_by_dtype_func))

        collection = {}
        for data_type in data_types:
            collection[data_type] = collection_by_dtype_func(data_type)

        return collection

    # TODO. Here we should process sentences and also output the updated results.
    def __iter_words_embedded_vectors(self, predefined_embedding):
        embedding_mapper = EmbeddedTermMapping(predefined_embedding=predefined_embedding)
        for pnc in self.__pncs.itervalues():
            assert(isinstance(pnc, ParsedNewsCollection))
            for news_ID in pnc.iter_news_ids():
                for word, embedding in embedding_mapper.iter_mapped(pnc.iter_news_terms(news_id=news_ID)):
                    yield word, embedding

    # endregion
