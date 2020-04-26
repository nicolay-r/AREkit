import logging

from arekit.common.frames.collection import FramesCollection
from arekit.common.model.labeling.paired import PairedLabelsHelper
from arekit.common.model.labeling.single import SingleLabelsHelper
from arekit.common.synonyms import SynonymsCollection
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection

from arekit.contrib.experiments.base import BaseExperiment
from arekit.contrib.experiments.single.embedding.entities import generate_entity_embeddings
from arekit.contrib.experiments.single.embedding.frames import init_frames_embedding
from arekit.contrib.experiments.single.opinions import extract_text_opinions
from arekit.contrib.experiments.single.embedding.tokens import create_tokens_embedding
from arekit.contrib.experiments.single.embedding.words import init_custom_words_embedding
from arekit.contrib.experiments.single.helpers.bags import BagsCollectionHelper
from arekit.contrib.experiments.single.helpers.text_opinions import LabeledLinkedTextOpinionCollectionHelper
from arekit.contrib.experiments.utils import create_input_sample
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig

from arekit.networks.context.training.bags.collection import BagsCollection
from arekit.networks.context.embedding.input import create_term_embedding_matrix
from arekit.common.data_type import DataType


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

        self.__synonyms = experiment.DataIO.SynonymsCollection

        self.__labels_helper = SingleLabelsHelper() if config.ClassesCount == 3 else PairedLabelsHelper()

        self.__text_opinion_collections = self.__create_collection(
            lambda data_type: extract_text_opinions(experiment=experiment,
                                                    data_type=data_type,
                                                    terms_per_context=config.TermsPerContext))

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

        self.__bags_collection = self.__create_collection(
            lambda data_type: self.create_bags_collection(
                text_opinions_collection=self.__text_opinion_collections[data_type],
                frames_collection=experiment.DataIO.FramesCollection,
                synonyms_collection=self.__synonyms,
                data_type=data_type,
                config=config))

        self.__bags_collection_helpers = self.__create_collection(
            lambda data_type: BagsCollectionHelper(bags_collection=self.__bags_collection[data_type],
                                                   name=data_type))

        self.__text_opinion_collection_helpers = self.__create_collection(
            lambda data_type: LabeledLinkedTextOpinionCollectionHelper(
                collection=self.__text_opinion_collections[data_type],
                labels_helper=self.__labels_helper,
                name=data_type))

        norm, _ = self.__text_opinion_collection_helpers[DataType.Train].get_statistic()
        config.set_class_weights(norm)

        config.notify_initialization_completed()

    # region Properties

    @property
    def BagsCollections(self):
        return self.__bags_collection

    @property
    def BagsCollectionHelpers(self):
        return self.__bags_collection_helpers

    @property
    def TextOpinionCollections(self):
        return self.__text_opinion_collections

    @property
    def TextOpinionCollectionHelpers(self):
        return self.__text_opinion_collection_helpers

    @property
    def LabelsHelper(self):
        return self.__labels_helper

    # endregion

    @staticmethod
    def create_bags_collection(text_opinions_collection,
                               frames_collection,
                               synonyms_collection,
                               data_type,
                               config):
        assert(isinstance(text_opinions_collection, LabeledLinkedTextOpinionCollection))
        assert(isinstance(frames_collection, FramesCollection))
        assert(isinstance(synonyms_collection, SynonymsCollection))
        assert(isinstance(config, DefaultNetworkConfig))

        collection = BagsCollection.from_linked_text_opinions(
            text_opinions_collection,
            data_type=data_type,
            bag_size=config.BagSize,
            shuffle=True,
            create_empty_sample_func=None,
            create_sample_func=lambda r: create_input_sample(
                text_opinion=r,
                frames_collection=frames_collection,
                synonyms_collection=synonyms_collection,
                config=config))

        return collection

    # region private methods

    @staticmethod
    def __create_collection(collection_by_data_type_func):
        assert(callable(collection_by_data_type_func))

        collection = {}
        for data_type in DataType.iter_supported():
            collection[data_type] = collection_by_data_type_func(data_type)

        return collection

    def __iter_all_parsed_collections(self):
        for collection in self.__text_opinion_collections.itervalues():
            assert(isinstance(collection, LabeledLinkedTextOpinionCollection))
            yield collection.RelatedParsedNewsCollection

    def __iter_all_terms(self, term_check_func):
        for pnc in self.__iter_all_parsed_collections():
            assert(isinstance(pnc, ParsedNewsCollection))
            for news_ID in pnc.iter_news_ids():
                for term in pnc.iter_news_terms(news_ID):
                    if not term_check_func(term):
                        continue
                    yield term

    # endregion
