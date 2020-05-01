#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging

from arekit.common.experiment.base import BaseExperiment
from arekit.common.experiment.neutral.algo.default import DefaultNeutralAnnotationAlgorithm
from arekit.common.experiment.neutral.annot.base import BaseNeutralAnnotator
from arekit.common.experiment.data_type import DataType

logger = logging.getLogger(__name__)


class ThreeScaleNeutralAnnotator(BaseNeutralAnnotator):
    """
    Neutral Annotator for RuSentRel Collection (of each data_type)

    Three scale classification task.
    """

    IGNORED_ENTITY_VALUES = [u"author", u"unknown"]

    def __init__(self):
        super(ThreeScaleNeutralAnnotator, self).__init__(
            annot_name=u"neutral_3_scale")
        self.__algo = None

    # region private methods

    def __create_opinions_for_extraction(self, doc_id, data_type):
        assert(isinstance(self.Experiment, BaseExperiment))
        news, _ = self.Experiment.read_parsed_news(doc_id=doc_id)
        opinions = self.Experiment.read_etalon_opinion_collection(doc_id=doc_id)
        collection = self.__algo.make_neutrals(
            news_id=doc_id,
            entities_collection=news.DocEntities,
            sentiment_opinions=opinions if data_type == DataType.Train else None)

        return collection

    # endregion

    def initialize(self, experiment):
        assert(isinstance(experiment, BaseExperiment))
        super(ThreeScaleNeutralAnnotator, self).initialize(experiment)

        self.__algo = DefaultNeutralAnnotationAlgorithm(
            synonyms=experiment.DataIO.SynonymsCollection,
            create_parsed_news_func=lambda doc_id: self.Experiment.read_parsed_news(doc_id=doc_id),
            iter_news_ids=self.iter_doc_ids_to_compare(),
            ignored_entity_values=self.IGNORED_ENTITY_VALUES)

    def create_collection(self, data_type):
        assert(isinstance(data_type, unicode))

        filtered_iter = self.filter_non_created_doc_ids(data_type=data_type,
                                                        all_doc_ids=self.iter_doc_ids_to_compare())

        for doc_id, filepath in filtered_iter:
            logger.debug("Create Neutral File (MODE {}): '{}'".format(data_type, filepath))
            collection = self.__create_opinions_for_extraction(doc_id=doc_id,
                                                               data_type=data_type)

            self.Experiment.DataIO.OpinionFormatter.save_to_file(collection=collection,
                                                                 filepath=filepath)



