#!/usr/bin/python
# -*- coding: utf-8 -*-
import collections

from core.evaluation.cmp_opinions import OpinionCollectionsToCompare
from core.evaluation.evaluators import metrics
from core.evaluation.evaluators.base import BaseEvaluator
from core.evaluation.labels import PositiveLabel, NegativeLabel, Label
from core.evaluation.results.base import DocumentCompareTable
from core.evaluation.results.two_class import TwoClassEvalResult
from core.common.opinions.collection import OpinionCollection


class TwoClassEvaluator(BaseEvaluator):

    def __init__(self, synonyms):
        super(TwoClassEvaluator, self).__init__(synonyms=synonyms)
        self.__pos_label = PositiveLabel()
        self.__neg_label = NegativeLabel()

    def calc_a_file(self, cmp_pair):
        assert(isinstance(cmp_pair, OpinionCollectionsToCompare))

        cmp_table = self.calc_difference(etalon_opins=cmp_pair.EtalonOpinionCollection,
                                         test_opins=cmp_pair.TestOpinionCollection)

        return cmp_table, \
               self.__has_opinions_with_label(cmp_pair.EtalonOpinionCollection, self.__pos_label), \
               self.__has_opinions_with_label(cmp_pair.TestOpinionCollection, self.__neg_label)

    @staticmethod
    def __has_opinions_with_label(opinions, label):
        assert(isinstance(label, Label))
        assert(isinstance(opinions, OpinionCollection))
        for opinion in opinions:
            if opinion.sentiment.to_int() == label.to_int():
                return True
        return False

    def evaluate(self, cmp_pairs):
        assert(isinstance(cmp_pairs, collections.Iterable))

        result = TwoClassEvalResult()
        for cmp_pair in cmp_pairs:
            cmp_table, has_pos, has_neg = self.calc_a_file(cmp_pair)

            pos_prec, pos_recall = metrics.calc_prec_and_recall(cmp_table=cmp_table,
                                                                label=self.__pos_label,
                                                                opinions_exist=has_pos)

            neg_prec, neg_recall = metrics.calc_prec_and_recall(cmp_table=cmp_table,
                                                                label=self.__neg_label,
                                                                opinions_exist=has_neg)

            result.add_document_results(doc_id=cmp_pair.DocumentID,
                                        cmp_table=DocumentCompareTable(cmp_table),
                                        pos_recall=pos_recall,
                                        neg_recall=neg_recall,
                                        pos_prec=pos_prec,
                                        neg_prec=neg_prec)

        result.calculate()
        return result
