#!/usr/bin/python
# -*- coding: utf-8 -*-
from core.evaluation.evaluators.base import BaseEvaluator
from core.evaluation.labels import PositiveLabel, NegativeLabel, Label
from core.evaluation.results.two_class import TwoClassEvalResult
from core.source.opinion import OpinionCollection


class TwoClassEvaluator(BaseEvaluator):

    def __init__(self, synonyms):
        super(TwoClassEvaluator, self).__init__(synonyms=synonyms)
        self.__pos_label = PositiveLabel()
        self.__neg_label = NegativeLabel()

    def calc_a_file(self, files_to_compare, debug):
        test_opins, etalon_opins = super(TwoClassEvaluator, self).calc_a_file(files_to_compare=files_to_compare,
                                                                              debug=debug)
        results = self.calc_difference(etalon_opins, test_opins)

        return results, \
               self.__has_opinions_with_label(etalon_opins, self.__pos_label),\
               self.__has_opinions_with_label(etalon_opins, self.__neg_label)

    @staticmethod
    def __has_opinions_with_label(opinions, label):
        assert(isinstance(label, Label))
        assert(isinstance(opinions, OpinionCollection))
        for opinion in opinions:
            if opinion.sentiment.to_int() == label.to_int():
                return True
        return False

    def evaluate(self, files_to_compare_list, debug=False):
        assert(isinstance(files_to_compare_list, list))

        result = TwoClassEvalResult()
        for files_to_compare in files_to_compare_list:

            # TODO. Save cmp into results.

            cmp_results, has_pos, has_neg = self.calc_a_file(files_to_compare, debug=debug)

            # TODO. Maybe in metrics method calc_prec_and_recall
            pos_prec, pos_recall = self.calc_prec_and_recall(results=cmp_results,
                                                             label=self.__pos_label,
                                                             opinions_exist=has_pos)

            neg_prec, neg_recall = self.calc_prec_and_recall(results=cmp_results,
                                                             label=self.__neg_label,
                                                             opinions_exist=has_neg)

            result.add_document_results(doc_id=files_to_compare.index,
                                        pos_recall=pos_recall,
                                        neg_recall=neg_recall,
                                        pos_prec=pos_prec,
                                        neg_prec=neg_prec)

        result.calculate()
        return result
