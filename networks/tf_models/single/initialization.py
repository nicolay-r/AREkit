import logging

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.opinions import extract_text_opinions
from arekit.common.experiment.labeling import LabeledCollection
from arekit.common.model.helpers.text_opinions import LabeledLinkedTextOpinionCollectionHelper
from arekit.common.model.labeling.single import SingleLabelsHelper
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.sample import InputSample
from arekit.networks.embedding.input import create_term_embedding_matrix
from arekit.networks.tf_models.single.embedding.entities import generate_entity_embeddings
from arekit.networks.tf_models.single.embedding.frames import init_frames_embedding
from arekit.networks.tf_models.single.embedding.tokens import create_tokens_embedding
from arekit.networks.tf_models.single.embedding.words import init_custom_words_embedding
from arekit.networks.tf_models.single.helpers.bags import BagsCollectionHelper
from arekit.networks.training.bags.collection.single import SingleBagsCollection

logger = logging.getLogger(__name__)


class SingleInstanceModelExperimentInitializer(object):

    def __init__(self, experiment, config):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(config, DefaultNetworkConfig))

        word_embedding = experiment.DataIO.WordEmbedding

        word_embedding.set_stemmer(experiment.DataIO.Stemmer)
        config.set_word_embedding(word_embedding)

        entity_embeddings = generate_entity_embeddings(
            use_types=config.UseEntityTypesInEmbedding,
            word_embedding=word_embedding)

        self.__pncs = self.__create_collection(
            lambda data_type: experiment.create_parsed_collection(data_type))

        self.__synonyms = experiment.DataIO.SynonymsCollection

        self.__labels_scaler = experiment.DataIO.LabelsScaler
        self.__labels_helper = SingleLabelsHelper(label_scaler=self.__labels_scaler)

        self.__text_opinion_helpers = self.__create_collection(
            lambda data_type: TextOpinionHelper(parsed_news_collection=self.__pncs[data_type])
        )

        self.__text_opinion_collections = self.__create_collection(
            lambda data_type: extract_text_opinions(experiment=experiment,
                                                    data_type=data_type,
                                                    terms_per_context=config.TermsPerContext,
                                                    iter_doc_ids=self.__pncs[data_type].iter_news_ids(),
                                                    text_opinion_helper=self.__text_opinion_helpers[data_type]))

        self.__labeled_collections = self.__create_collection(
            lambda data_type: LabeledCollection(collection=self.__text_opinion_collections[data_type])
        )

        custom_embedding = init_custom_words_embedding(iter_all_terms_func=self.__iter_all_terms,
                                                       entity_embeddings=entity_embeddings,
                                                       word_embedding=word_embedding,
                                                       config=config)

        config.set_custom_words_embedding(custom_embedding)

        token_embedding = create_tokens_embedding(word_embedding.VectorSize)
        config.set_token_embedding(token_embedding)

        frame_embedding = init_frames_embedding(iter_all_terms_func=self.__iter_all_terms,
                                                word_embedding=word_embedding)

        config.set_frames_embedding(frame_embedding)

        term_embedding = create_term_embedding_matrix(
                word_embedding=config.WordEmbedding,
                custom_embedding=config.CustomWordEmbedding,
                token_embedding=config.TokenEmbedding,
                frame_embedding=config.FrameEmbedding)
        config.set_term_embedding(term_embedding)

        self.__frames_collection = experiment.DataIO.FramesCollection

        self.__bags_collection = self.__create_collection(
            lambda data_type: self.create_bags_collection(
                data_type=data_type,
                config=config))

        self.__bags_collection_helpers = self.__create_collection(
            lambda data_type: BagsCollectionHelper(bags_collection=self.__bags_collection[data_type],
                                                   name=data_type))

        self.__text_opinion_collection_helpers = self.__create_collection(
            lambda data_type: LabeledLinkedTextOpinionCollectionHelper(
                collection=self.__text_opinion_collections[data_type],
                labels_helper=self.__labels_helper,
                text_opinion_helper=self.__text_opinion_helpers[data_type],
                name=data_type))

        norm, _ = self.__text_opinion_collection_helpers[DataType.Train].get_statistic()
        config.set_class_weights(norm)

        config.notify_initialization_completed()

    # region Properties

    @property
    def _LabelsScaler(self):
        return self.__labels_scaler

    @property
    def BagsCollections(self):
        return self.__bags_collection

    @property
    def BagsCollectionHelpers(self):
        return self.__bags_collection_helpers

    @property
    def LabeledCollection(self):
        return self.__labeled_collections

    @property
    def TextOpinionCollectionHelpers(self):
        return self.__text_opinion_collection_helpers

    @property
    def LabelsHelper(self):
        return self.__labels_helper

    @property
    def _BagCollectionType(self):
        return SingleBagsCollection

    # endregion

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
            create_sample_func=lambda text_opinion: self.__create_input_sample(text_opinion=text_opinion,
                                                                               config=config,
                                                                               data_type=data_type))

        return collection

    # region private methods

    def __create_input_sample(self, text_opinion, data_type, config):
        """
        Creates an input for Neural Network model
        """
        assert(isinstance(data_type, unicode))
        assert(isinstance(config, DefaultNetworkConfig))
        assert(self.__text_opinion_helpers[data_type].check_ends_has_same_sentence_index(text_opinion))

        return InputSample.from_text_opinion(
            text_opinion=text_opinion,
            config=config,
            frames_collection=self.__frames_collection,
            synonyms_collection=self.__synonyms,
            label_scaler=self.__labels_scaler,
            text_opinion_helper=self.__text_opinion_helpers[data_type])

    @staticmethod
    def __create_collection(collection_by_data_type_func):
        assert(callable(collection_by_data_type_func))

        collection = {}
        for data_type in DataType.iter_supported():
            collection[data_type] = collection_by_data_type_func(data_type)

        return collection

    # TODO. Simplify (parsed_news already provides filter)
    def __iter_all_terms(self, term_check_func):
        for pnc in self.__pncs.itervalues():
            assert(isinstance(pnc, ParsedNewsCollection))
            for news_ID in pnc.iter_news_ids():
                for term in pnc.iter_news_terms(news_ID):
                    if not term_check_func(term):
                        continue
                    yield term

    # endregion
