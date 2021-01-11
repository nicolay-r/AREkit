import collections

from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.results.two_class import TwoClassEvalResult
from arekit.common.labels.base import PositiveLabel, NegativeLabel, Label
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.evaluation.cmp_opinions import OpinionCollectionsToCompare
from arekit.common.evaluation.evaluators import metrics
from arekit.common.evaluation.evaluators.base import BaseEvaluator


class TwoClassEvaluator(BaseEvaluator):

    def __init__(self, eval_mode=EvaluationModes.Extraction):
        super(TwoClassEvaluator, self).__init__(eval_mode=eval_mode)
        self.__pos_label = PositiveLabel()
        self.__neg_label = NegativeLabel()

    # region private methods

    def __calc_for_pair(self, cmp_pair):
        assert(isinstance(cmp_pair, OpinionCollectionsToCompare))

        cmp_table = self.calc_difference(etalon_opins=cmp_pair.EtalonOpinionCollection,
                                         test_opins=cmp_pair.TestOpinionCollection)

        return cmp_table, \
               self.__has_opinions_with_label(cmp_pair.EtalonOpinionCollection, self.__pos_label), \
               self.__has_opinions_with_label(cmp_pair.EtalonOpinionCollection, self.__neg_label)

    @staticmethod
    def __has_opinions_with_label(opinions, label):
        assert(isinstance(label, Label))
        assert(isinstance(opinions, OpinionCollection))
        for opinion in opinions:
            if opinion.Sentiment == label:
                return True
        return False

    # endregion

    def evaluate(self, cmp_pairs):
        assert(isinstance(cmp_pairs, collections.Iterable))

        result = TwoClassEvalResult()
        for cmp_pair in cmp_pairs:
            cmp_table, has_pos, has_neg = self.__calc_for_pair(cmp_pair)

            pos_prec, pos_recall = metrics.calc_prec_and_recall(cmp_table=cmp_table,
                                                                label=self.__pos_label,
                                                                opinions_exist=has_pos)

            neg_prec, neg_recall = metrics.calc_prec_and_recall(cmp_table=cmp_table,
                                                                label=self.__neg_label,
                                                                opinions_exist=has_neg)

            result.add_document_results(doc_id=cmp_pair.DocumentID,
                                        cmp_table=cmp_table,
                                        pos_recall=pos_recall,
                                        neg_recall=neg_recall,
                                        pos_prec=pos_prec,
                                        neg_prec=neg_prec)

        result.calculate()
        return result
