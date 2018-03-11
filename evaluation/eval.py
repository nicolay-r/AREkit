#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd

from core.evaluation.labels import PositiveLabel, NegativeLabel, NeutralLabel
from core.evaluation.statistic import FilesToCompare
from core.processing.stemmer import Stemmer
from core.source.opinion import OpinionCollection


class Evaluator:

    # Columns
    C_POS_PREC = 'pos_prec'
    C_NEG_PREC = 'neg_prec'
    C_POS_RECALL = 'pos_recall'
    C_NEG_RECALL = 'neg_recall'
    C_F1_POS = 'f1_pos'
    C_F1_NEG = 'f1_neg'
    C_F1 = 'f1'

    # Columns for differences
    C_WHO = 'who'
    C_TO = 'to'
    C_ORIG = 'how_orig'
    C_RES = 'how_results'
    C_CMP = 'comparison'

    def __init__(self, synonyms_filepath, user_answers_filepath):
        self.synonyms_filepath = synonyms_filepath
        self.user_answers = user_answers_filepath
        self.stemmer = Stemmer()

        self.pos = PositiveLabel()
        self.neg = NegativeLabel()
        self.neu = NeutralLabel()

    @staticmethod
    def get_result_columns():
        return [Evaluator.C_POS_PREC,
                Evaluator.C_POS_PREC,
                Evaluator.C_NEG_PREC,
                Evaluator.C_POS_RECALL,
                Evaluator.C_NEG_RECALL,
                Evaluator.C_F1_POS,
                Evaluator.C_F1_NEG,
                Evaluator.C_F1]

    @staticmethod
    def _calcRecall(results, answers, label):
        assert(isinstance(label, PositiveLabel) or isinstance(label, NegativeLabel))
        if len(results[results[Evaluator.C_ORIG] == label.to_str()]) != 0:
            return 1.0 * len(answers[(answers[Evaluator.C_CMP] == True)]) / len(results[results[Evaluator.C_ORIG] == label.to_str()])
        else:
            return 0.0

    @staticmethod
    def _calcPrecision(answers):
        if len(answers) != 0:
            return 1.0 * len(answers[(answers[Evaluator.C_CMP] == True)]) / len(answers)
        else:
            return 0.0

    def _calcPrecisionAndRecall(self, results):
        """ Расчет полноты и точности.
        """
        pos_answers = results[(results[Evaluator.C_RES] == self.pos.to_str())]
        neg_answers = results[(results[Evaluator.C_RES] == self.neg.to_str())]

        pos_prec = self._calcPrecision(pos_answers)
        neg_prec = self._calcPrecision(neg_answers)

        pos_recall = self._calcRecall(results, pos_answers, self.pos)
        neg_recall = self._calcRecall(results, neg_answers, self.neg)

        assert(type(pos_prec) == float)
        assert(type(neg_prec) == float)
        assert(type(pos_recall) == float)
        assert(type(neg_recall) == float)

        return pos_prec, neg_prec, pos_recall, neg_recall

    def _check(self, etalon_opins, test_opins):
        assert(isinstance(etalon_opins, OpinionCollection))
        assert(isinstance(test_opins, OpinionCollection))

        df = pd.DataFrame(
                columns=[self.C_WHO, self.C_TO, self.C_ORIG, self.C_RES, self.C_CMP])

        r_ind = 0
        # Append everithing that exist in etalon collection.
        for o_etalon in etalon_opins:
            comparison = False
            has_opinion = test_opins.has_opinion_by_synonyms(o_etalon)

            if has_opinion:
                o_test = test_opins.get_opinion_by_synonyms(o_etalon)
                comparison = o_test.sentiment == o_etalon.sentiment

            df.loc[r_ind] = [o_etalon.value_left.encode('utf-8'),
                             o_etalon.value_right.encode('utf-8'),
                             o_etalon.sentiment.to_str(),
                             None if not has_opinion else o_test.sentiment.to_str(),
                             comparison]
            r_ind += 1

        # Append everithing that exist in test collection.
        for o_test in test_opins:
            has_opinion = etalon_opins.has_opinion_by_synonyms(o_test)
            if has_opinion:
                continue
            df.loc[r_ind] = [o_test.value_left.encode('utf-8'),
                             o_test.value_right.encode('utf-8'),
                             None,
			                 o_test.sentiment.to_str(),
                             False]
            r_ind += 1

        return df

    # TODO. change it with the list of FilesToCompare objects.
    def _calc_a_file(self, files_to_compare):
        assert(isinstance(files_to_compare, FilesToCompare))

        # Reading test answers.
        test_opins = OpinionCollection.from_file(
                files_to_compare.test_filepath, self.synonyms_filepath)

        # Reading etalon answers.
        etalon_opins = OpinionCollection.from_file(
                files_to_compare.etalon_filepath, self.synonyms_filepath)

        # print "{} <-> {}, {}".format(
        #         files_to_compare.test_filepath,
        #         files_to_compare.etalon_filepath,
        #         files_to_compare.index)

        # Comparing test and etalon results.
        results = self._check(etalon_opins, test_opins)

        # Save result comparison into file.
        # TODO. remove path declaration from here.
        comparison_file = "{}/art{}.comp.txt".format(
                self.user_answers, str(files_to_compare.index))
        results.to_csv(comparison_file)

        return self._calcPrecisionAndRecall(results)

    def evaluate(self, files_to_compare_list):
        """ Main evaluation subprogram
        """
        assert(type(files_to_compare_list) == list)

        pos_prec, neg_prec, pos_recall, neg_recall = (0, 0, 0, 0)

        for files_to_compare in files_to_compare_list:
            [pos_prec1, neg_prec1, pos_recall1, neg_recall1] = self._calc_a_file(files_to_compare)

            pos_prec += pos_prec1
            neg_prec += neg_prec1
            pos_recall += pos_recall1
            neg_recall += neg_recall1

        # print len(files_to_compare_list)

        pos_prec /= len(files_to_compare_list)
        neg_prec /= len(files_to_compare_list)
        pos_recall /= len(files_to_compare_list)
        neg_recall /= len(files_to_compare_list)

        if pos_prec * pos_recall != 0:
            f1_pos = 2 * pos_prec * pos_recall / (pos_prec + pos_recall)
        else:
            f1_pos = 0

        if neg_prec * neg_recall != 0:
            f1_neg = 2 * neg_prec * neg_recall / (neg_prec + neg_recall)
        else:
            f1_neg = 0

        return {self.C_POS_PREC: pos_prec,
                self.C_NEG_PREC: neg_prec,
                self.C_POS_RECALL: pos_recall,
                self.C_NEG_RECALL: neg_recall,
                self.C_F1_POS: f1_pos,
                self.C_F1_NEG: f1_neg,
                self.C_F1: (f1_pos + f1_neg) / 2}
