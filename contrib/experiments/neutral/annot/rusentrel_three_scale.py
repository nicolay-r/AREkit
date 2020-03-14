#!/usr/bin/python
# -*- coding: utf-8 -*-
import utils
import logging

from arekit.common.labels.base import NeutralLabel
from arekit.contrib.experiments.io_utils_base import BaseExperimentsIOUtils
from arekit.contrib.experiments.neutral.algo.default import DefaultNeutralAnnotationAlgorithm
from arekit.contrib.experiments.neutral.annot.rusentrel_two_scale import RuSentRelTwoScaleNeutralAnnotator
from arekit.networks.data_type import DataType
from arekit.processing.lemmatization.base import Stemmer
from arekit.source.rusentrel.helpers.parsed_news import RuSentRelParsedNewsHelper
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from arekit.source.rusentrel.news import RuSentRelNews
from arekit.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from arekit.source.rusentrel.opinions.opinion import RuSentRelOpinion


logger = logging.getLogger(__name__)


class RuSentRelThreeScaleNeutralAnnotator(RuSentRelTwoScaleNeutralAnnotator):
    """
    Neutral Annotator for RuSentRel Collection (of each data_type)

    Three scale classification task.
    """

    __annot_name = u"neutral_3_scale"
    IGNORED_ENTITY_VALUES = [u"author", u"unknown"]

    def __init__(self, experiments_io, stemmer, create_synonyms_collection):
        assert(isinstance(experiments_io, BaseExperimentsIOUtils))
        assert(isinstance(stemmer, Stemmer))
        assert(callable(create_synonyms_collection))

        super(RuSentRelThreeScaleNeutralAnnotator).__init__(
            experiments_io=experiments_io,
            create_synonyms_collection=create_synonyms_collection)

        self.__stemmer = stemmer

        self.__algo = DefaultNeutralAnnotationAlgorithm(
            synonyms=self.__synonyms,
            create_opinion_func=lambda s_value, t_value: RuSentRelOpinion(
                value_source=s_value,
                value_target=t_value,
                sentiment=NeutralLabel()),
            create_opinion_collection_func=lambda: RuSentRelOpinionCollection(opinions=None,
                                                                              synonyms=self.SynonoymsCollection),
            create_parsed_news_func=lambda doc_id: self.__create_parsed_news(doc_id=doc_id,
                                                                             synonyms=self.SynonoymsCollection,
                                                                             stemmer=self.__stemmer),
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

    # endregion

    def create(self, data_type):
        assert(isinstance(data_type, unicode))

        for doc_id in RuSentRelIOUtils.iter_collection_indices():

            neutral_filepath = self.get_opin_filepath(doc_id=doc_id,
                                                      data_type=data_type,
                                                      output_dir=self.ExperimentsIO.get_experiments_dir())

            if utils.check_file_already_exsited(filepath=neutral_filepath, logger=logger):
                continue

            msg = "Create Neutral File (MODE {}): '{}'".format(data_type, neutral_filepath)

            logger.debug(msg)

            entities = RuSentRelDocumentEntityCollection.read_collection(
                doc_id=doc_id,
                synonyms=self.__synonyms)

            news = RuSentRelNews.read_document(doc_id=doc_id, entities=entities)
            opinions = RuSentRelOpinionCollection.read_collection(doc_id=doc_id, synonyms=self.SynonoymsCollection)

            neutral_opins = self.__algo.make_neutrals(
                news_id=doc_id,
                entities_collection=news.DocEntities,
                sentiment_opinions=opinions if data_type == DataType.Train else None)

            neutral_opins.save_to_file(neutral_filepath)



