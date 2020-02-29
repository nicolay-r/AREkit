#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
from os.path import join

from arekit.common.labels.base import NeutralLabel
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.contrib.experiments.io_utils_base import BaseExperimentsIO
from arekit.contrib.experiments.neutral_annot.default import DefaultNeutralAnnotationAlgorithm
from arekit.contrib.experiments.utils import get_path_of_subfolder_in_experiments_dir
from arekit.source.rusentrel.helpers.parsed_news import RuSentRelParsedNewsHelper
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from arekit.source.rusentrel.news import RuSentRelNews
from arekit.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from arekit.source.rusentrel.opinions.opinion import RuSentRelOpinion
from arekit.source.rusentrel.synonyms import RuSentRelSynonymsCollection
from arekit.processing.lemmatization.mystem import MystemWrapper


class RuSentRelNeutralAnnotatorIO(object):
    """
    Neutral Annotator for RuSentRel Collection
    """

    IGNORED_ENTITY_VALUES = [u"author", u"unknown"]

    def __init__(self, experiments_io):
        assert(isinstance(experiments_io, BaseExperimentsIO))

        self.__experiments_io = experiments_io
        self.__stemmer = MystemWrapper()
        self.__synonyms = RuSentRelSynonymsCollection.read_collection(
            stemmer=self.__stemmer,
            is_read_only=True)

        self.__pnc = ParsedNewsCollection()
        self.__init_parsed_news_collection()

        self.__algo = DefaultNeutralAnnotationAlgorithm(
            synonyms=self.__synonyms,
            create_opinion_func=lambda s_value, t_value: RuSentRelOpinion(
                value_source=s_value,
                value_target=t_value,
                sentiment=NeutralLabel()),
            create_opinion_collection_func=lambda: RuSentRelOpinionCollection(opinions=None, synonyms=self.__synonyms),
            ignored_entity_values=self.IGNORED_ENTITY_VALUES)

    # region private methods

    def __init_parsed_news_collection(self):
        for doc_id in RuSentRelIOUtils.iter_collection_indices():
            entities = RuSentRelDocumentEntityCollection.read_collection(doc_id=doc_id,
                                                                         synonyms=self.__synonyms)

            news = RuSentRelNews.read_document(doc_id=doc_id, entities=entities)

            parsed_news = RuSentRelParsedNewsHelper.create_parsed_news(rusentrel_news_id=doc_id,
                                                                       rusentrel_news=news,
                                                                       keep_tokens=False,
                                                                       stemmer=self.__stemmer)
            self.__pnc.add(parsed_news)

    # endregion

    def create(self, is_train):
        assert(isinstance(is_train, bool))

        for doc_id in RuSentRelIOUtils.iter_collection_indices():

            neutral_filepath = self.get_rusentrel_neutral_opin_filepath(doc_id=doc_id,
                                                                        is_train=is_train,
                                                                        experiments_io=self.__experiments_io)

            # Skip if this file is already exists
            if os.path.isfile(neutral_filepath):
                if os.path.getsize(neutral_filepath):
                    # TODO. To logger.
                    print "Skipping File: {} [OK. File already exists]".format(neutral_filepath)
                    continue

            # TODO. To logger.
            print "Create Neutral File (MODE {}): '{}'".format("TRAIN" if is_train else "TEST",
                                                               neutral_filepath)

            entities = RuSentRelDocumentEntityCollection.read_collection(
                doc_id=doc_id,
                synonyms=self.__synonyms)

            news = RuSentRelNews.read_document(doc_id=doc_id, entities=entities)
            opinions = RuSentRelOpinionCollection.read_collection(doc_id=doc_id, synonyms=self.__synonyms)

            neutral_opins = self.__algo.make_neutrals(
                news_id=doc_id,
                entities_collection=news.DocEntities,
                sentiment_opinions=opinions if is_train else None)

            neutral_opins.save_to_file(neutral_filepath)

    @staticmethod
    def get_rusentrel_neutral_opin_filepath(doc_id, is_train, experiments_io, model_name=u"universal"):
        assert(isinstance(doc_id, int))
        assert(isinstance(is_train, bool))
        assert(isinstance(model_name, unicode))
        assert(isinstance(experiments_io, BaseExperimentsIO))

        root = get_path_of_subfolder_in_experiments_dir(subfolder_name=model_name,
                                                        experiments_io=experiments_io)
        return join(root, u"art{}.neut.{}.txt".format(doc_id, u'train' if is_train else u'test'))


