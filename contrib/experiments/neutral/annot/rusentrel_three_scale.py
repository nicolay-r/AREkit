#!/usr/bin/python
# -*- coding: utf-8 -*-
import utils
import logging

from arekit.contrib.experiments.data_io import DataIO
from arekit.contrib.experiments.neutral.algo.default import DefaultNeutralAnnotationAlgorithm
from arekit.contrib.experiments.neutral.annot.rusentrel_two_scale import RuSentRelTwoScaleNeutralAnnotator
from arekit.networks.data_type import DataType
from arekit.processing.lemmatization.base import Stemmer
from arekit.source.rusentrel.helpers.parsed_news import RuSentRelParsedNewsHelper
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from arekit.source.rusentrel.news import RuSentRelNews
from arekit.source.rusentrel.opinions.collection import RuSentRelOpinionCollection

logger = logging.getLogger(__name__)


class RuSentRelThreeScaleNeutralAnnotator(RuSentRelTwoScaleNeutralAnnotator):
    """
    Neutral Annotator for RuSentRel Collection (of each data_type)

    Three scale classification task.
    """

    __annot_name = u"neutral_3_scale"
    IGNORED_ENTITY_VALUES = [u"author", u"unknown"]

    def __init__(self, data_io, stemmer):
        assert(isinstance(data_io, DataIO))
        assert(isinstance(stemmer, Stemmer))

        super(RuSentRelThreeScaleNeutralAnnotator).__init__(data_io=data_io)

        self.__synonyms = data_io.SynonymsCollection

        self.__algo = DefaultNeutralAnnotationAlgorithm(
            synonyms=self.__synonyms,
            create_parsed_news_func=lambda doc_id: self.__create_parsed_news(doc_id=doc_id,
                                                                             synonyms=self.__synonyms,
                                                                             stemmer=stemmer),
            iter_news_ids=RuSentRelIOUtils.iter_collection_indices(),
            ignored_entity_values=self.IGNORED_ENTITY_VALUES)

    @property
    def AnnotationModelName(self):
        return self.__annot_name

    # region private methods

    @staticmethod
    def __create_parsed_news(doc_id, synonyms, stemmer):
        entities = RuSentRelDocumentEntityCollection.read_collection(doc_id=doc_id,
                                                                     synonyms=synonyms)

        news = RuSentRelNews.read_document(doc_id=doc_id,
                                           entities=entities)

        return RuSentRelParsedNewsHelper.create_parsed_news(rusentrel_news_id=doc_id,
                                                            rusentrel_news=news,
                                                            keep_tokens=False,
                                                            stemmer=stemmer)

    def __create_opinions_for_extraction(self, doc_id, data_type):
        entities = RuSentRelDocumentEntityCollection.read_collection(doc_id=doc_id,
                                                                     synonyms=self.__synonyms)

        news = RuSentRelNews.read_document(doc_id=doc_id, entities=entities)
        opinions = RuSentRelOpinionCollection.load_collection(doc_id=doc_id,
                                                              synonyms=self.__synonyms)

        collection = self.__algo.make_neutrals(
            news_id=doc_id,
            entities_collection=news.DocEntities,
            sentiment_opinions=opinions if data_type == DataType.Train else None)

        return collection

    # endregion

    def create(self, data_type):
        assert(isinstance(data_type, unicode))

        filtered_iter = self._filter_non_created_doc_ids(data_type=data_type,
                                                         all_doc_ids=RuSentRelIOUtils.iter_collection_indices())

        for doc_id, filepath in filtered_iter:
            logger.debug("Create Neutral File (MODE {}): '{}'".format(data_type, filepath))
            collection = self.__create_opinions_for_extraction(doc_id=doc_id,
                                                               data_type=data_type)
            self.DataIO.OpinionFormatter.save_to_file(collection=collection,
                                                      filepath=filepath)



