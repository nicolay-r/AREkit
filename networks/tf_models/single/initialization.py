import collections
import logging

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.opinions import extract_text_opinions
from arekit.common.experiment.labeling import LabeledCollection
from arekit.common.model.labeling.single import SingleLabelsHelper
from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.news.parsed.collection import ParsedNewsCollection
from arekit.common.text_opinions.text_opinion import TextOpinion

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.sample import InputSample

from arekit.networks.input.encoder import NetworkInputEncoder
from arekit.networks.input.terms_mapping import EmbeddedTermMapping
from arekit.networks.tf_models.single.helpers.bags import BagsCollectionHelper
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

        # TODO. This should not be used since we already have a connections.
        # TODO. This should be removed.
        self.__text_opinion_helpers = self.__create_collection(
            data_types=supported_data_types,
            collection_by_dtype_func=lambda data_type: TextOpinionHelper(parsed_news_collection=self.__pncs[data_type]))

        # TODO. This should not be used since we already have a connections.
        # TODO. This should be removed.
        self.__text_opinion_collections = self.__create_collection(
            supported_data_types,
            # TODO. This should not be used since we already have a tsv with extracted text opinions.
            lambda data_type: extract_text_opinions(experiment=experiment,
                                                    data_type=data_type,
                                                    terms_per_context=config.TermsPerContext,
                                                    iter_doc_ids=self.__pncs[data_type].iter_news_ids(),
                                                    text_opinion_helper=self.__text_opinion_helpers[data_type]))

        # TODO: Samples labeling collection
        self.__labeled_collections = self.__create_collection(
            supported_data_types,
            # TODO. Labeled collection will be simplified
            lambda data_type: LabeledCollection(collection=self.__text_opinion_collections[data_type]))

        self.__frames_collection = experiment.DataIO.FramesCollection

        # TODO. We assume here to iterate over tsv records.
        self.__bags_collection = self.__create_collection(
            supported_data_types,
            # TODO. We assume here to iterate over tsv records.
            lambda data_type: self.create_bags_collection(
                data_type=data_type,
                config=config))

        self.__bags_collection_helpers = self.__create_collection(
            supported_data_types,
            lambda data_type: BagsCollectionHelper(bags_collection=self.__bags_collection[data_type],
                                                   name=data_type))

        norm, _ = self.get_statistic(collection=self.__text_opinion_collections[DataType.Train],
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
    def TextOpitnionCollections(self):
        return self.__text_opinion_collections

    @property
    def BagsCollections(self):
        return self.__bags_collection

    @property
    def BagsCollectionHelpers(self):
        return self.__bags_collection_helpers

    @property
    def LabeledCollection(self):
        # TODO. Samples labeling collection
        return self.__labeled_collections

    # TODO. Remove
    @property
    def LabelsHelper(self):
        return self.__labels_helper

    @property
    def TextOpinionHelpers(self):
        return self.__text_opinion_helpers

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

    def create_bags_collection(self, data_type, config):
        assert(isinstance(config, DefaultNetworkConfig))

        collection = self._BagCollectionType.from_linked_text_opinions(
            self.__text_opinion_collections[data_type],
            data_type=data_type,
            bag_size=config.BagSize,
            shuffle=True,
            create_empty_sample_func=self._create_empty_sample_func,
            text_opinion_helper=self.__text_opinion_helpers[data_type],
            # TODO. We assume here to iterate over tsv records.
            create_sample_func=lambda text_opinion: self.__create_input_sample(text_opinion=text_opinion,
                                                                               config=config,
                                                                               data_type=data_type))

        return collection

    # region private methods

    # TODO. We assume here to iterate over tsv records.
    def __create_input_sample(self, text_opinion, data_type, config):
        """
        Creates an input for Neural Network model
        """
        assert(isinstance(data_type, DataType))
        assert(isinstance(config, DefaultNetworkConfig))
        assert(self.__text_opinion_helpers[data_type].check_ends_has_same_sentence_index(text_opinion))

        # TODO. We assume here to iterate over tsv records.
        return InputSample.from_text_opinion(
            text_opinion=text_opinion,
            config=config,
            frames_collection=self.__frames_collection,
            synonyms_collection=self.__synonyms,
            label_scaler=self.__labels_scaler,
            text_opinion_helper=self.__text_opinion_helpers[data_type],
            string_entity_formatter=self.__string_entity_formatter)

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
