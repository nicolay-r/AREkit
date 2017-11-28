#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd

from core.processing.stemmer import Stemmer
from core.source.opinion import OpinionCollection
from core.labels import PositiveLabel, NegativeLabel, NeutralLabel


class Evaluator:

    # Columns
    C_POS_PREC = 'pos_prec'
    C_NEG_PREC = 'neg_prec'
    C_POS_RECALL = 'pos_recall'
    C_NEG_RECALL = 'neg_recall'
    C_F1_POS = 'f1_pos'
    C_F1_NEG = 'f1_neg'
    C_F1 = 'f1'

    def __init__(self, synonyms_filepath, user_answers_filepath, etalon_filepath):
        self.synonyms_filepath = synonyms_filepath
        self.user_answers = user_answers_filepath
        self.etalon_answers = etalon_filepath
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


    def _calcPrecisionAndRecall(self, results):
        """ Расчет полноты и точности.
        """
        # Берем все позитивные и негативные ответы команд
        # TODO. Constants in different file
        pos_answers = results[(results['how_results'] == self.pos.to_str())]
        neg_answers = results[(results['how_results'] == self.neg.to_str())]

        # Расчет точности.
        if len(pos_answers) != 0:
            pos_prec = 1.0 * len(pos_answers[(pos_answers['comparison'] == True)]) / len(pos_answers)
        else:
            pos_prec = 0.0
        if len(neg_answers) != 0:
            neg_prec = 1.0 * len(neg_answers[(neg_answers['comparison'] == True)]) / len(neg_answers)
        else:
            neg_prec = 0.0

        # Расчет полноты.
        if len(results[results['how_orig'] == self.pos.to_str()]) != 0:
            pos_recall = 1.0 * len(pos_answers[(pos_answers['comparison'] == True)]) / len(results[results['how_orig'] == self.pos.to_str()])
        else:
            pos_recall = 0.0
        if len(results[results['how_orig'] == self.neg.to_str()]) != 0:
            neg_recall = 1.0 * len(neg_answers[(neg_answers['comparison'] == True)]) / len(results[results['how_orig'] == self.neg.to_str()])
        else:
            neg_recall = 0.0

        assert(type(pos_prec) == float)
        assert(type(neg_prec) == float)
        assert(type(pos_recall) == float)
        assert(type(neg_recall) == float)

        return pos_prec, neg_prec, pos_recall, neg_recall

    def _check(self, etalon_opins, test_opins):
        assert(isinstance(etalon_opins, OpinionCollection))
        assert(isinstance(test_opins, OpinionCollection))

        df = pd.DataFrame(
            columns=['who', 'to', 'how_orig', 'how_results', 'comparison'])

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

    def _calc_a_file(self, num):
        """ Data calculation for a file of 'num' index
        """
        print "Reading {}".format(num)
        # Reading test answers.
        test_filepath = "{}/art{}.opin.txt".format(self.user_answers, str(num))
        test_opins = OpinionCollection.from_file(
                test_filepath, self.synonyms_filepath)

        # Reading etalon answers.
        experts_filepath = "{}/art{}.opin.txt".format(self.etalon_answers, str(num))
        etalon_opins = OpinionCollection.from_file(
                experts_filepath, self.synonyms_filepath)

        # Comparing test and etalon results.
        results = self._check(etalon_opins, test_opins)

        # Save result comparison into file.
        comparison_file = "{}/art{}.comp.txt".format(self.user_answers, str(num))
        results.to_csv(comparison_file)

        return self._calcPrecisionAndRecall(results)

    def evaluate(self, test_indices):
        """ Main evaluation subprogram
        """
        pos_prec, neg_prec, pos_recall, neg_recall = (0, 0, 0, 0)

        for n in test_indices:
            [pos_prec1, neg_prec1, pos_recall1, neg_recall1] = self._calc_a_file(n)

            pos_prec += pos_prec1
            neg_prec += neg_prec1
            pos_recall += pos_recall1
            neg_recall += neg_recall1

        pos_prec /= len(test_indices)
        neg_prec /= len(test_indices)
        pos_recall /= len(test_indices)
        neg_recall /= len(test_indices)

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
