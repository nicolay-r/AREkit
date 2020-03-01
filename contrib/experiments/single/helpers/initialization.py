import numpy as np
import itertools
import collections

from arekit.common import utils
from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.frames.collection import FramesCollection
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.common.labels.base import NeutralLabel

from arekit.contrib.experiments.single.helpers.bags import BagsCollectionHelper
from arekit.contrib.experiments.single.helpers.parsed_news import ParsedNewsHelper
from arekit.contrib.experiments.single.helpers.text_opinions import LabeledLinkedTextOpinionCollectionHelper
from arekit.contrib.experiments.sources.rusentrel_io import RuSentRelBasedExperimentIO
from arekit.networks.data_type import DataType
from arekit.networks.context.debug import DebugKeys
from arekit.contrib.networks.sample import InputSample
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.networks.context.training.bags.collection import BagsCollection
from arekit.networks.context.embedding import entity
from arekit.networks.context.embedding.input import create_term_embedding_matrix
from arekit.networks.labeling.paired import PairedLabelsHelper
from arekit.networks.labeling.single import SingleLabelsHelper
from arekit.source.embeddings.rusvectores import RusvectoresEmbedding
from arekit.common.embeddings.tokens import TokenEmbedding
from arekit.source.ruattitudes.helpers.linked_text_opinions import RuAttitudesNewsTextOpinionExtractorHelper
from arekit.source.ruattitudes.news import RuAttitudesNews
from arekit.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.source.rusentiframes.helpers.parse import RuSentiFramesParseHelper
from arekit.source.rusentrel.helpers.linked_text_opinions import RuSentRelNewsTextOpinionExtractorHelper
from arekit.source.rusentrel.news import RuSentRelNews
from arekit.common.opinions.base import Opinion
from arekit.common.embeddings.base import Embedding
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.common.text_opinions.base import TextOpinion
from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection


# TODO. Rename as ...Initializer
class SingleInstanceModelInitHelper(object):

    CAPITAL_ENTITY_TYPE = u"CAPITAL"
    STATE_ENTITY_TYPE = u"STATE"

    def __init__(self, io, config):
        assert(isinstance(io, RuSentRelBasedExperimentIO))
        assert(isinstance(config, DefaultNetworkConfig))

        print "Loading word embedding: {}".format(io.get_word_embedding_filepath())
        word_embedding = RusvectoresEmbedding.from_word2vec_format(
            filepath=io.get_word_embedding_filepath(),
            binary=True)
        word_embedding.set_stemmer(config.Stemmer)
        config.set_word_embedding(word_embedding)

        self. __log_states_presented = 0
        self. __log_capitals_presented = 0

        self.__entity_embeddings = self.__generate_entity_embeddings(
            use_types=config.UseEntityTypesInEmbedding,
            word_embedding=word_embedding)

        self.__synonyms = io.SynonymsCollection

        self.__frames_collection = RuSentiFramesCollection.read_collection()

        self.__capitals_list = io.get_capitals_list()

        self.__states_list = io.get_states_list()

        self.__frame_variants = FrameVariantsCollection.from_iterable(
            variants_with_id=self.__frames_collection.iter_frame_id_and_variants(),
            stemmer=config.Stemmer)

        self.__labels_helper = SingleLabelsHelper() if config.ClassesCount == 3 else PairedLabelsHelper()

        self.__text_opinion_collections = {
            DataType.Test: self.__extract_text_opinions(io=io,
                                                        data_type=DataType.Test,
                                                        config=config),
            DataType.Train: self.__extract_text_opinions(io=io,
                                                         data_type=DataType.Train,
                                                         config=config)
        }

        print "Replaced with 'STATES' entities: {}".format(self.__log_states_presented)
        print "Replaced with 'CAPITALS' entities: {}".format(self.__log_capitals_presented)

        # TODO. In separated file (WORDS)
        # Init Custom Words Embedding.
        custom_embedding = Embedding.from_list_with_embedding_func(
            words_iter=self.__iter_custom_words(config=config),
            embedding_func=lambda term: self.__custom_embedding_func(term, word_embedding=word_embedding))

        config.set_custom_words_embedding(custom_embedding)

        # TODO. In separated file. (TOKENS)
        # Init Token Embedding.
        seed_token_offset = 12345
        token_embedding = TokenEmbedding.from_supported_tokens(
            vector_size=word_embedding.VectorSize,
            random_vector_func=lambda size, t_ind: utils.get_random_normal_distribution(
                vector_size=size,
                seed=t_ind + seed_token_offset,
                loc=0.05,
                scale=0.025))

        config.set_token_embedding(token_embedding)

        # TODO. In separated file. (FRAMES)
        # Init Frame Embedding.
        frame_embedding = Embedding.from_list_with_embedding_func(
            words_iter=self.__iter_variants(),
            embedding_func=lambda variant_value: word_embedding.create_term_embedding(term=variant_value,
                                                                                      # TODO. The same parameter.
                                                                                      max_part_size=3)
        )
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
                self.__text_opinion_collections[DataType.Test],
                labels_helper=self.__labels_helper,
                name=DataType.Test),
            DataType.Train: LabeledLinkedTextOpinionCollectionHelper(
                self.__text_opinion_collections[DataType.Train],
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

    def __iter_all_entity_types(self):
        return itertools.chain(entity.iter_entity_types(),
                               [self.CAPITAL_ENTITY_TYPE, self.STATE_ENTITY_TYPE])

    @staticmethod
    def __mask_to_word(mask):
        if mask == entity.ANY_ENTITY_MASK:
            return u"e"
        elif mask == entity.OBJ_ENTITY_MASK:
            return u"object"
        elif mask == entity.SUBJ_ENTITY_MASK:
            return u"subject"

        return None

    @staticmethod
    def __entity_type_to_word(e_type):
        if e_type == entity.ORG_ENTITY_TYPE:
            return u"organization"
        if e_type == entity.LOC_ENTITY_TYPE:
            return u'location'
        if e_type == entity.PER_ENTITY_TYPE:
            return u'person'
        if e_type == entity.GEOPOLIT_ENTITY_TYPE:
            return u'political'
        if e_type == SingleInstanceModelInitHelper.CAPITAL_ENTITY_TYPE:
            return u'capital'
        if e_type == SingleInstanceModelInitHelper.STATE_ENTITY_TYPE:
            return u'state'

    def __generate_entity_embeddings(self, use_types, word_embedding):
        assert(isinstance(use_types, bool))
        assert(isinstance(word_embedding, RusvectoresEmbedding))

        # Unique start index
        embeddings = {}

        v_index = 0

        for e_mask in entity.iter_entity_masks():
            for e_type in self.__iter_all_entity_types():

                value = entity.compose_entity_mask(
                    e_mask=e_mask,
                    e_type=e_type if use_types else None)

                if value not in embeddings:

                    mask = self.__mask_to_word(e_mask)
                    t = self.__entity_type_to_word(e_type)

                    m_ind = word_embedding.try_find_index_by_word(mask, lemmatize=False)
                    t_ind = word_embedding.try_find_index_by_word(t, lemmatize=False)

                    e_v = np.mean([word_embedding.get_vector_by_index(m_ind),
                                   word_embedding.get_vector_by_index(t_ind)],
                                  axis=0)

                    embeddings[value] = e_v

                v_index += 1

        return embeddings

    @staticmethod
    def create_bags_collection(text_opinions_collection, frames_collection, synonyms_collection, data_type, config):
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
            create_sample_func=lambda r: SingleInstanceModelInitHelper.create_sample(text_opinion=r,
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

    # region private methods

    # TODO. In utils
    def __custom_embedding_func(self, term, word_embedding):
        assert(isinstance(term, unicode))

        if term in self.__entity_embeddings:
            return self.__entity_embeddings[term]

        # TODO. Entity has _ separator!!!
        return word_embedding.create_term_embedding(term)

    # TODO. In separated file (FRAMES)
    def __iter_variants(self):
        frame_variants_iter = self.__iter_all_terms(lambda t: isinstance(t, TextFrameVariant))
        for variant in frame_variants_iter:
            yield variant.Variant.get_value()

    # TODO. In separated file (WORDS)
    def __iter_custom_words(self, config):
        all_terms_iter = self.__iter_all_terms(lambda t:
                                               isinstance(t, unicode) and
                                               t not in config.WordEmbedding)

        for e_mask in entity.iter_entity_masks():
            for e_type in self.__iter_all_entity_types():

                # TODO. Entity has a different separator for type
                yield entity.compose_entity_mask(e_mask=e_mask, e_type=e_type)

                # TODO. Entity has a different separator for type
            yield entity.compose_entity_mask(e_mask=e_mask, e_type=None)

        for term in all_terms_iter:
            yield term

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

    # TODO. REMOVE
    @staticmethod
    def __find_or_create_reversed_opinion(opinion, opinion_collections):
        assert(isinstance(opinion, Opinion))
        assert(isinstance(opinion_collections, collections.Iterable))

        reversed_opinion = Opinion(source_value=opinion.TargetValue,
                                   target_value=opinion.SourceValue,
                                   sentiment=NeutralLabel())

        for collection in opinion_collections:
            if collection.has_synonymous_opinion(reversed_opinion):
                return collection.get_synonymous_opinion(reversed_opinion)

        return reversed_opinion

    # TODO. In separated file (TEXT OPINIONS)
    def __extract_text_opinions(self, io, data_type, config):
        assert(isinstance(io, RuSentRelBasedExperimentIO))
        assert(isinstance(data_type, unicode))
        assert(isinstance(config, DefaultNetworkConfig))

        parsed_collection = ParsedNewsCollection()
        text_opinions = LabeledLinkedTextOpinionCollection(parsed_news_collection=parsed_collection)
        for news_id in io.iter_data_indices(data_type):

            news, parsed_news = self.__read_document(io=io,
                                                     news_id=news_id,
                                                     config=config)

            parsed_news.modify_parsed_sentences(
                lambda sentence: RuSentiFramesParseHelper.parse_frames_in_parsed_text(
                    frame_variants_collection=self.__frame_variants,
                    parsed_text=sentence)
            )

            parsed_news.modify_entity_types(lambda value: self.__provide_entity_type_by_value(value))

            if not parsed_collection.contains_id(news_id):
                parsed_collection.add(parsed_news)
            else:
                print "Warning: Skipping document with id={}, news={}".format(news.DocumentID, news)

            opinions_it = self.__iter_opinion_collections(io=io,
                                                          news_id=news_id,
                                                          data_type=data_type)

            for opinions in opinions_it:
                self.__fill_text_opinions(text_opinions=text_opinions,
                                          news=news,
                                          opinions=opinions,
                                          terms_per_context=config.TermsPerContext)

        return text_opinions

    # TODO. In separated file (TEXT OPINIONS)
    def __provide_entity_type_by_value(self, value):

        if not self.__synonyms.contains_synonym_value(value):
            return None

        for s_value in self.__synonyms.iter_synonym_values(value):
            if s_value in self.__capitals_list:
                self.__log_capitals_presented += 1
                return self.CAPITAL_ENTITY_TYPE

        for s_value in self.__synonyms.iter_synonym_values(value):
            if s_value in self.__states_list:
                self.__log_states_presented += 1
                return self.STATE_ENTITY_TYPE

        return None

    # TODO. In separated file (TEXT OPINIONS)
    @staticmethod
    def __read_document(io, news_id, config):
        assert(isinstance(news_id, int))
        assert(isinstance(config, DefaultNetworkConfig))

        news, parsed_news = io.read_parsed_news(doc_id=news_id,
                                                keep_tokens=config.KeepTokens,
                                                stemmer=config.Stemmer)

        if DebugKeys.NewsTermsStatisticShow:
            ParsedNewsHelper.debug_statistics(parsed_news)
        if DebugKeys.NewsTermsShow:
            ParsedNewsHelper.debug_show_terms(parsed_news)

        return news, parsed_news

    # TODO. In separated file (TEXT OPINIONS)
    @staticmethod
    def __iter_opinion_collections(io, news_id, data_type):
        assert(isinstance(news_id, int))
        assert(isinstance(data_type, unicode))

        neutral = io.read_neutral_opinion_collection(doc_id=news_id,
                                                     data_type=data_type)

        if neutral is not None:
            yield neutral

        if data_type == DataType.Train:
            yield io.read_etalon_opinion_collection(doc_id=news_id)

    # TODO. In separated file (TEXT OPINIONS)
    def __fill_text_opinions(self,
                             text_opinions,
                             news,
                             opinions,
                             terms_per_context):
        assert(isinstance(news, RuSentRelNews) or isinstance(news, RuAttitudesNews))
        assert(isinstance(text_opinions, LabeledLinkedTextOpinionCollection))
        assert(isinstance(opinions, OpinionCollection))
        assert(isinstance(terms_per_context, int))

        def __check_text_opinion(text_opinion):
            assert(isinstance(text_opinion, TextOpinion))
            return InputSample.check_ability_to_create_sample(
                window_size=terms_per_context,
                text_opinion=text_opinion)

        if isinstance(news, RuSentRelNews):
            return RuSentRelNewsTextOpinionExtractorHelper.add_entries(
                text_opinion_collection=text_opinions,
                news=news,
                opinions=opinions,
                check_text_opinion_is_correct=__check_text_opinion)

        elif isinstance(news, RuAttitudesNews):
            return RuAttitudesNewsTextOpinionExtractorHelper.add_entries(
                text_opinion_collection=text_opinions,
                news=news,
                check_text_opinion_is_correct=__check_text_opinion)

    # endregion
