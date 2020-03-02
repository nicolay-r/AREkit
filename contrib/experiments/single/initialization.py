import logging

from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.frames.collection import FramesCollection
from arekit.common.synonyms import SynonymsCollection
from arekit.contrib.experiments.single.embedding.entities import generate_entity_embeddings
from arekit.contrib.experiments.single.embedding.frames import init_frames_embedding
from arekit.contrib.experiments.single.embedding.opinions import extract_text_opinions
from arekit.contrib.experiments.single.embedding.tokens import create_tokens_embedding
from arekit.contrib.experiments.single.embedding.words import init_custom_words_embedding

from arekit.contrib.experiments.single.helpers.bags import BagsCollectionHelper
from arekit.contrib.experiments.single.helpers.text_opinions import LabeledLinkedTextOpinionCollectionHelper
from arekit.contrib.experiments.sources.rusentrel_io import RuSentRelBasedExperimentIO
from arekit.networks.data_type import DataType
from arekit.contrib.networks.sample import InputSample
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.networks.context.training.bags.collection import BagsCollection
from arekit.networks.context.embedding.input import create_term_embedding_matrix
from arekit.networks.labeling.paired import PairedLabelsHelper
from arekit.networks.labeling.single import SingleLabelsHelper
from arekit.source.embeddings.rusvectores import RusvectoresEmbedding
from arekit.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.common.text_opinions.base import TextOpinion
from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection


logger = logging.getLogger(__name__)


class SingleInstanceModelInitializer(object):

    def __init__(self, io, config):
        assert(isinstance(io, RuSentRelBasedExperimentIO))
        assert(isinstance(config, DefaultNetworkConfig))

        logger.info("Loading word embedding: {}".format(io.get_word_embedding_filepath()))

        word_embedding = RusvectoresEmbedding.from_word2vec_format(
            filepath=io.get_word_embedding_filepath(),
            binary=True)
        word_embedding.set_stemmer(config.Stemmer)
        config.set_word_embedding(word_embedding)

        self.__entity_embeddings = generate_entity_embeddings(
            use_types=config.UseEntityTypesInEmbedding,
            word_embedding=word_embedding)

        self.__synonyms = io.SynonymsCollection

        self.__frames_collection = RuSentiFramesCollection.read_collection()

        self.__labels_helper = SingleLabelsHelper() if config.ClassesCount == 3 else PairedLabelsHelper()

        frame_variants = FrameVariantsCollection.from_iterable(
            variants_with_id=self.__frames_collection.iter_frame_id_and_variants(),
            stemmer=config.Stemmer)

        self.__text_opinion_collections = {
            DataType.Test: extract_text_opinions(io=io,
                                                 data_type=DataType.Test,
                                                 frame_variants_collection=frame_variants,
                                                 config=config),
            DataType.Train: extract_text_opinions(io=io,
                                                  data_type=DataType.Train,
                                                  frame_variants_collection=frame_variants,
                                                  config=config)
        }

        custom_embedding = init_custom_words_embedding(m_init=self,
                                                       word_embedding=word_embedding,
                                                       config=config)

        config.set_custom_words_embedding(custom_embedding)

        token_embedding = create_tokens_embedding(word_embedding.VectorSize)
        config.set_token_embedding(token_embedding)

        frame_embedding = init_frames_embedding(m_init=self,
                                                word_embedding=word_embedding)

        config.set_frames_embedding(frame_embedding)

        term_embedding = create_term_embedding_matrix(
                word_embedding=config.WordEmbedding,
                custom_embedding=config.CustomWordEmbedding,
                token_embedding=config.TokenEmbedding,
                frame_embedding=config.FrameEmbedding)

        config.set_term_embedding(term_embedding)

        self.__bags_collection = {
            DataType.Test: self.create_bags_collection(
                text_opinions_collection=self.__text_opinion_collections[DataType.Test],
                frames_collection=self.__frames_collection,
                synonyms_collection=self.__synonyms,
                data_type=DataType.Test,
                config=config),
            DataType.Train: self.create_bags_collection(
                text_opinions_collection=self.__text_opinion_collections[DataType.Train],
                frames_collection=self.__frames_collection,
                synonyms_collection=self.__synonyms,
                data_type=DataType.Train,
                config=config)
        }

        self.__bags_collection_helpers = {
            DataType.Train: BagsCollectionHelper(bags_collection=self.__bags_collection[DataType.Train],
                                                 name=DataType.Train),
            DataType.Test: BagsCollectionHelper(bags_collection=self.__bags_collection[DataType.Test],
                                                name=DataType.Test)
        }

        self.__text_opinion_collection_helpers = {
            DataType.Test: LabeledLinkedTextOpinionCollectionHelper(
                collection=self.__text_opinion_collections[DataType.Test],
                labels_helper=self.__labels_helper,
                name=DataType.Test),
            DataType.Train: LabeledLinkedTextOpinionCollectionHelper(
                collection=self.__text_opinion_collections[DataType.Train],
                labels_helper=self.__labels_helper,
                name=DataType.Train)
        }

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

    # region public methods

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
            create_sample_func=lambda r: SingleInstanceModelInitializer.create_sample(
                text_opinion=r,
                frames_collection=frames_collection,
                synonyms_collection=synonyms_collection,
                config=config))

        return collection

    @staticmethod
    def create_sample(text_opinion, frames_collection, synonyms_collection, config):
        assert(isinstance(text_opinion, TextOpinion))
        assert(TextOpinionHelper.check_ends_has_same_sentence_index(text_opinion))

        text_opinion_collection = text_opinion.Owner
        assert(isinstance(text_opinion_collection, LabeledLinkedTextOpinionCollection))

        parsed_news_collection = text_opinion_collection.RelatedParsedNewsCollection
        assert(isinstance(parsed_news_collection, ParsedNewsCollection))

        return InputSample.from_text_opinion(
            text_opinion=text_opinion,
            parsed_news=parsed_news_collection.get_by_news_id(text_opinion.NewsID),
            config=config,
            frames_collection=frames_collection,
            synonyms_collection=synonyms_collection)

    # endregion

    def iter_all_parsed_collections(self):
        for collection in self.__text_opinion_collections.itervalues():
            assert(isinstance(collection, LabeledLinkedTextOpinionCollection))
            yield collection.RelatedParsedNewsCollection

    def iter_all_terms(self, term_check_func):
        for pnc in self.iter_all_parsed_collections():
            assert(isinstance(pnc, ParsedNewsCollection))
            for news_ID in pnc.iter_news_ids():
                for term in pnc.iter_news_terms(news_ID):
                    if not term_check_func(term):
                        continue
                    yield term
