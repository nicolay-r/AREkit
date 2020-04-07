#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging

from arekit.contrib.experiments.experiment_io import BaseExperimentNeuralNetworkIO
from arekit.contrib.experiments.neutral.algo.default import DefaultNeutralAnnotationAlgorithm
from arekit.contrib.experiments.neutral.annot.base import BaseNeutralAnnotator
from arekit.networks.data_type import DataType

logger = logging.getLogger(__name__)


class ThreeScaleNeutralAnnotator(BaseNeutralAnnotator):
    """
    Neutral Annotator for RuSentRel Collection (of each data_type)

    Three scale classification task.
    """

    IGNORED_ENTITY_VALUES = [u"author", u"unknown"]

    def __init__(self):
        super(ThreeScaleNeutralAnnotator).__init__(
            annot_name=u"neutral_3_scale")
        self.__algo = None
        self.__experiments_io = None

    # region private methods

    def __create_opinions_for_extraction(self, doc_id, data_type):
        assert(isinstance(self.__experiments_io, BaseExperimentNeuralNetworkIO))
        news, _ = self.__experiments_io.read_parsed_news(doc_id=doc_id)
        opinions = self.__experiments_io.read_etalon_opinion_collection(doc_id=doc_id)
        collection = self.__algo.make_neutrals(
            news_id=doc_id,
            entities_collection=news.DocEntities,
            sentiment_opinions=opinions if data_type == DataType.Train else None)

        return collection

    # endregion

    def initialize(self, experiment_io):
        assert(isinstance(experiment_io, BaseExperimentNeuralNetworkIO))
        super(ThreeScaleNeutralAnnotator, self).initialize(experiment_io)

        self.__algo = DefaultNeutralAnnotationAlgorithm(
            synonyms=experiment_io.DataIO.SynonymsCollection,
            create_parsed_news_func=lambda doc_id: self.__experiments_io.read_parsed_news(doc_id=doc_id),
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

            self.ExperimentIO.DataIO.OpinionFormatter.save_to_file(collection=collection,
                                                                   filepath=filepath)



