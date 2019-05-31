#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd

from core.evaluation.labels import PositiveLabel, NegativeLabel, Label
from core.evaluation.result import EvalResult
from core.evaluation.utils import FilesToCompare
from core.source.opinion import OpinionCollection
from core.source.synonyms import SynonymsCollection


# TODO: New evaluator that is irrespective of the label
class Evaluator:

    # Columns for differences
    C_WHO = 'who'
    C_TO = 'to'
    C_ORIG = 'how_orig'
    C_RES = 'how_results'
    C_CMP = 'comparison'

    def __init__(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection) and synonyms.IsReadOnly)
        self.__synonyms = synonyms
        self.__pos_label = PositiveLabel()
        self.__neg_label = NegativeLabel()

    @staticmethod
    def __calc_recall(results, answers, label, answer_exist):
        assert(isinstance(results, pd.DataFrame))
        assert(isinstance(label, PositiveLabel) or isinstance(label, NegativeLabel))
        assert(isinstance(answer_exist, bool))
        total = len(results[results[Evaluator.C_ORIG] == label.to_str()])
        if total != 0:
            return 1.0 * len(answers[(answers[Evaluator.C_CMP] == True)]) / total
        else:
            return 0.0 if answer_exist else 1.0

    @staticmethod
    def __calc_precision(answers, answer_exist):
        assert(isinstance(answers, pd.DataFrame))
        assert(isinstance(answer_exist, bool))
        total = len(answers)
        if total != 0:
            return 1.0 * len(answers[(answers[Evaluator.C_CMP] == True)]) / total
        else:
            return 0.0 if answer_exist else 1.0

    def __calc_prec_and_recall(self, results, label, opinions_exist):
        assert(isinstance(opinions_exist, bool))
        assert(isinstance(label, Label))

        answers = results[(results[Evaluator.C_RES] == label.to_str())]
        p = self.__calc_precision(answers, answer_exist=opinions_exist)
        r = self.__calc_recall(results, answers, label, answer_exist=opinions_exist)

        assert(isinstance(p, float))
        assert(isinstance(r, float))

        return p, r

    def __calc_difference(self, etalon_opins, test_opins):
        assert(isinstance(etalon_opins, OpinionCollection))
        assert(isinstance(test_opins, OpinionCollection))

        df = pd.DataFrame(
                columns=[self.C_WHO, self.C_TO, self.C_ORIG, self.C_RES, self.C_CMP])

        r_ind = 0
        for o_etalon in etalon_opins:
            comparison = False
            has_opinion = test_opins.has_synonymous_opinion(o_etalon)

            # TODO: Disable this check in case of non label evaluation method.

            if has_opinion:
                o_test = test_opins.get_synonymous_opinion(o_etalon)
                comparison = o_test.sentiment == o_etalon.sentiment

            df.loc[r_ind] = [o_etalon.value_left.encode('utf-8'),
                             o_etalon.value_right.encode('utf-8'),
                             o_etalon.sentiment.to_str(),
                             None if not has_opinion else o_test.sentiment.to_str(),
                             comparison]
            r_ind += 1

        for o_test in test_opins:
            has_opinion = etalon_opins.has_synonymous_opinion(o_test)
            if has_opinion:
                continue
            df.loc[r_ind] = [o_test.value_left.encode('utf-8'),
                             o_test.value_right.encode('utf-8'),
                             None, o_test.sentiment.to_str(), False]
            r_ind += 1

        return df

    def __calc_a_file(self, files_to_compare, debug):
        assert(isinstance(files_to_compare, FilesToCompare))

        # Reading test answers.
        test_opins = OpinionCollection.from_file(
            filepath=files_to_compare.TestFilepath,
            synonyms=self.__synonyms)

        # Reading etalon answers.
        etalon_opins = OpinionCollection.from_file(
            filepath=files_to_compare.EtalonFilepath,
            synonyms=self.__synonyms)

        if debug:
            print "{} <-> {}, {}".format(
                    files_to_compare.TestFilepath,
                    files_to_compare.EtalonFilepath,
                    files_to_compare.index)

        results = self.__calc_difference(etalon_opins, test_opins)

        return results, \
               self.__has_opinions_with_label(etalon_opins, PositiveLabel()),\
               self.__has_opinions_with_label(etalon_opins, NegativeLabel())

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

        result = EvalResult()
        for files_to_compare in files_to_compare_list:
            cmp_results, has_pos, has_neg = self.__calc_a_file(files_to_compare, debug=debug)

            pos_prec, pos_recall = self.__calc_prec_and_recall(results=cmp_results,
                                                               label=self.__pos_label,
                                                               opinions_exist=has_pos)

            neg_prec, neg_recall = self.__calc_prec_and_recall(results=cmp_results,
                                                               label=self.__neg_label,
                                                               opinions_exist=has_neg)

            result.add_document_results(doc_id=files_to_compare.index,
                                        pos_recall=pos_recall,
                                        neg_recall=neg_recall,
                                        pos_prec=pos_prec,
                                        neg_prec=neg_prec)

        result.calculate()
        return result
