#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

from arekit.common.entities.base import Entity
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.entities.collection import EntityCollection
from arekit.common.labels.base import NeutralLabel
from arekit.contrib.experiments.sources.rusentrel_neutrals_io import RuSentRelNeutralIOUtils
from arekit.source.rusentrel.helpers.parsed_news import RuSentRelParsedNewsHelper
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from arekit.source.rusentrel.news import RuSentRelNews
from arekit.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from arekit.source.rusentrel.opinions.opinion import RuSentRelOpinion
from arekit.source.rusentrel.synonyms import RuSentRelSynonymsCollection
from arekit.processing.lemmatization.mystem import MystemWrapper


class RuSentRelNeutralAnnotationCreator:

    IGNORED_ENTITY_VALUES = [
        u"author",
        u"unknown"]

    def __init__(self):
        self.__stemmer = MystemWrapper()
        self.__synonyms = RuSentRelSynonymsCollection.read_collection(stemmer=self.__stemmer,
                                                                      is_read_only=True)
        self.__pnc = ParsedNewsCollection()

        for doc_id in RuSentRelIOUtils.iter_collection_indices():
            entities = RuSentRelDocumentEntityCollection.read_collection(doc_id=doc_id,
                                                                         synonyms=self.__synonyms)

            news = RuSentRelNews.read_document(doc_id=doc_id, entities=entities)

            parsed_news = RuSentRelParsedNewsHelper.create_parsed_news(rusentrel_news_id=doc_id,
                                                                       rusentrel_news=news,
                                                                       keep_tokens=False,
                                                                       stemmer=self.__stemmer)
            self.__pnc.add(parsed_news)

    def create(self, is_train):
        assert(isinstance(is_train, bool))

        for doc_id in RuSentRelIOUtils.iter_collection_indices():

            neutral_filepath = RuSentRelNeutralIOUtils.get_rusentrel_neutral_opin_filepath(doc_id=doc_id,
                                                                                           is_train=is_train)

            # Skip if this file is already exists
            if os.path.isfile(neutral_filepath):
                if os.path.getsize(neutral_filepath):
                    print "Skipping File: {} [OK. File already exists]".format(neutral_filepath)
                    continue

            print "Create Neutral File (MODE {}): '{}'".format("TRAIN" if is_train else "TEST",
                                                               neutral_filepath)

            entities = RuSentRelDocumentEntityCollection.read_collection(
                doc_id=doc_id,
                synonyms=self.__synonyms)

            news = RuSentRelNews.read_document(doc_id=doc_id, entities=entities)
            opinions = RuSentRelOpinionCollection.read_collection(doc_id=doc_id, synonyms=self.__synonyms)

            neutral_opins = self.__make_neutrals(
                n_id=doc_id,
                entities_collection=news.DocEntities,
                sentiment_opinions=opinions if is_train else None)

            neutral_opins.save_to_file(neutral_filepath)

    # TODO. To NeutralAnnotatorAlgorithm

    # region private methods

    # TODO. To NeutralAnnotatorAlgorithm
    @staticmethod
    def __is_ignored_entity_value(entity_value):
        assert(isinstance(entity_value, unicode))
        return entity_value in RuSentRelNeutralAnnotationCreator.IGNORED_ENTITY_VALUES

    # TODO. To NeutralAnnotatorAlgorithm
    @staticmethod
    def __create_key_by_entity_pair(e1, e2):
        assert(isinstance(e1, Entity))
        assert(isinstance(e2, Entity))
        return u"{}_{}".format(e1.IdInDocument, e2.IdInDocument)

    # TODO. To NeutralAnnotatorAlgorithm
    def __get_distance_in_terms_between_entities(self, n_id, e1, e2):
        assert(isinstance(e1, Entity))
        assert(isinstance(e2, Entity))

        nt = self.__pnc.get_by_news_id(n_id)

        assert(isinstance(nt, ParsedNews))

        erp1 = nt.get_entity_document_level_term_index(e1.IdInDocument)
        erp2 = nt.get_entity_document_level_term_index(e2.IdInDocument)
        return abs(erp1 - erp2)

    # TODO. To NeutralAnnotatorAlgorithm
    def __get_distance_in_sentences_between_entities(self, n_id, e1, e2):
        assert(isinstance(e1, Entity))
        assert(isinstance(e2, Entity))

        nt = self.__pnc.get_by_news_id(n_id)

        assert(isinstance(nt, ParsedNews))

        e1_ind = nt.get_entity_sentence_index(e1.IdInDocument)
        e2_ind = nt.get_entity_sentence_index(e2.IdInDocument)
        return abs(e1_ind - e2_ind)

    # TODO. To NeutralAnnotatorAlgorithm
    def __create_opinions_between_entities(self, relevant_pairs, entities_collection):
        assert(isinstance(entities_collection, EntityCollection))
        assert(self.__synonyms.IsReadOnly is True)

        extracted_count = 0
        neutral_opinions = RuSentRelOpinionCollection(opinions=None, synonyms=self.__synonyms)

        for e1 in entities_collection:
            assert(isinstance(e1, Entity))

            for e2 in entities_collection:
                assert(isinstance(e2, Entity))

                key = RuSentRelNeutralAnnotationCreator.__create_key_by_entity_pair(e1=e1, e2=e2)
                if key not in relevant_pairs:
                    continue

                opinion = RuSentRelOpinion(e1.Value, e2.Value, NeutralLabel())

                if neutral_opinions.has_synonymous_opinion(opinion):
                    continue

                neutral_opinions.add_opinion(opinion)
                extracted_count += 1

        print "Neutral opinions extracted: {}".format(extracted_count)

        return neutral_opinions

    # TODO. To NeutralAnnotatorAlgorithm
    def __make_neutrals(self,
                        n_id,
                        entities_collection,
                        sentiment_opinions=None):
        assert(isinstance(n_id, int))
        assert(isinstance(entities_collection, EntityCollection))

        relevant_pairs = {}

        for e1 in entities_collection:
            assert(isinstance(e1, Entity))

            for e2 in entities_collection:
                assert(isinstance(e2, Entity))

                if e1.IdInDocument == e2.IdInDocument:
                    continue

                if RuSentRelNeutralAnnotationCreator.__is_ignored_entity_value(entity_value=e1.Value):
                    continue
                if RuSentRelNeutralAnnotationCreator.__is_ignored_entity_value(entity_value=e2.Value):
                    continue

                g1 = self.__synonyms.get_synonym_group_index(e1.Value)
                g2 = self.__synonyms.get_synonym_group_index(e2.Value)
                if g1 == g2:
                    continue

                s_dist = self.__get_distance_in_sentences_between_entities(n_id=n_id, e1=e1, e2=e2)

                if s_dist > 0:
                    continue

                t_dist = self.__get_distance_in_terms_between_entities(n_id=n_id, e1=e1, e2=e2)

                if t_dist > 10:
                    continue

                if sentiment_opinions is not None:
                    o = RuSentRelOpinion(e1.Value, e2.Value, NeutralLabel())
                    if sentiment_opinions.has_synonymous_opinion(opinion=o):
                        continue

                key = RuSentRelNeutralAnnotationCreator.__create_key_by_entity_pair(e1=e1, e2=e2)
                relevant_pairs[key] = 0

        opinions = self.__create_opinions_between_entities(
            relevant_pairs=relevant_pairs,
            entities_collection=entities_collection)

        return opinions

    # endregion
