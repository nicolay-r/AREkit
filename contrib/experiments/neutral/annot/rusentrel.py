#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import os
from os.path import join

from arekit.common.labels.base import NeutralLabel
from arekit.contrib.experiments.io_utils_base import BaseExperimentsIOUtils
from arekit.contrib.experiments.neutral.algo.default import DefaultNeutralAnnotationAlgorithm
from arekit.contrib.experiments.neutral.annot.base import BaseAnnotator
from arekit.contrib.experiments.utils import get_path_of_subfolder_in_experiments_dir
from arekit.networks.data_type import DataType
from arekit.source.rusentrel.helpers.parsed_news import RuSentRelParsedNewsHelper
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from arekit.source.rusentrel.news import RuSentRelNews
from arekit.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from arekit.source.rusentrel.opinions.opinion import RuSentRelOpinion
from arekit.source.rusentrel.synonyms import RuSentRelSynonymsCollection
from arekit.processing.lemmatization.mystem import MystemWrapper


logger = logging.getLogger(__name__)


class RuSentRelNeutralAnnotator(BaseAnnotator):
    """
    Neutral Annotator for RuSentRel Collection (of each data_type)
    Three scale classification task.
    """

    IGNORED_ENTITY_VALUES = [u"author", u"unknown"]

    def __init__(self, experiments_io):
        assert(isinstance(experiments_io, BaseExperimentsIOUtils))

        # TODO. Should be a parameters
        self.__experiments_io = experiments_io
        self.__stemmer = MystemWrapper()

        self.__synonyms = RuSentRelSynonymsCollection.read_collection(
            stemmer=self.__stemmer,
            is_read_only=True)

        self.__algo = DefaultNeutralAnnotationAlgorithm(
            synonyms=self.__synonyms,
            create_opinion_func=lambda s_value, t_value: RuSentRelOpinion(
                value_source=s_value,
                value_target=t_value,
                sentiment=NeutralLabel()),
            create_opinion_collection_func=lambda: RuSentRelOpinionCollection(opinions=None,
                                                                              synonyms=self.__synonyms),
            create_parsed_news_func=lambda doc_id: self.__create_parsed_news(doc_id=doc_id,
                                                                             synonyms=self.__synonyms,
                                                                             stemmer=self.__stemmer),
            iter_news_ids=RuSentRelIOUtils.iter_collection_indices(),
            ignored_entity_values=self.IGNORED_ENTITY_VALUES)

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

    @staticmethod
    def __data_type_to_string(data_type):
        if data_type == DataType.Train:
            return u'train'
        if data_type == DataType.Test:
            return u'test'

    # endregion

    def create(self, data_type):
        assert(isinstance(data_type, unicode))

        for doc_id in RuSentRelIOUtils.iter_collection_indices():

            neutral_filepath = self.get_opin_filepath(doc_id=doc_id,
                                                      data_type=data_type,
                                                      output_dir=self.__experiments_io.get_experiments_dir())

            # Skip if this file is already exists
            if os.path.isfile(neutral_filepath):
                if os.path.getsize(neutral_filepath):
                    logger.debug("Skipping File: {} [OK. File already exists]".format(neutral_filepath))
                    continue

            msg = "Create Neutral File (MODE {}): '{}'".format(data_type, neutral_filepath)

            logger.debug(msg)

            entities = RuSentRelDocumentEntityCollection.read_collection(
                doc_id=doc_id,
                synonyms=self.__synonyms)

            news = RuSentRelNews.read_document(doc_id=doc_id, entities=entities)
            opinions = RuSentRelOpinionCollection.read_collection(doc_id=doc_id, synonyms=self.__synonyms)

            neutral_opins = self.__algo.make_neutrals(
                news_id=doc_id,
                entities_collection=news.DocEntities,
                sentiment_opinions=opinions if data_type == DataType.Train else None)

            neutral_opins.save_to_file(neutral_filepath)

    def get_opin_filepath(self, doc_id, data_type, output_dir, model_name=u"universal"):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, unicode))
        assert(isinstance(model_name, unicode))
        assert(isinstance(output_dir, unicode))

        root = get_path_of_subfolder_in_experiments_dir(subfolder_name=model_name,
                                                        experiments_dir=output_dir)
        return join(root,
                    u"art{doc_id}.neut.{d_type}.txt".format(
                        doc_id=doc_id,
                        d_type=RuSentRelNeutralAnnotator.__data_type_to_string(data_type)))


