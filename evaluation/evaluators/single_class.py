from core.evaluation.evaluators.base import BaseEvaluator
from core.evaluation.labels import Label
from core.evaluation.results.base import DocumentCompareTable
from core.evaluation.results.single_class import SingleClassEvalResult
from core.source.opinion import OpinionCollection, Opinion
import metrics


class SentimentLabel(Label):

    def __init__(self):
        pass

    def to_str(self):
        return 'sent'

    def to_int(self):
        return int(3)

    def to_uint(self):
        return int(3)


class SingleClassEvaluator(BaseEvaluator):

    def __init__(self, synonyms):
        super(SingleClassEvaluator, self).__init__(synonyms=synonyms)
        self.__sentiment_label = SentimentLabel()

    def calc_a_file(self, files_to_compare, debug):
        test_opins, etalon_opins = super(SingleClassEvaluator, self).calc_a_file(
            files_to_compare=files_to_compare,
            debug=debug)

        assert(isinstance(test_opins, OpinionCollection))
        assert(isinstance(etalon_opins, OpinionCollection))

        test_opins = self.__clone_with_different_label(test_opins, self.__sentiment_label)
        etalon_opins = self.__clone_with_different_label(etalon_opins, self.__sentiment_label)

        results = self.calc_difference(etalon_opins, test_opins)

        return results

    def __clone_with_different_label(self, opinions, label):
        assert(isinstance(opinions, OpinionCollection))
        assert(isinstance(label, Label))

        ro = OpinionCollection(opinions=[],
                               synonyms=self.Synonyms)

        for o in opinions:
            assert(isinstance(o, Opinion))
            no = Opinion(value_left=o.value_left,
                         value_right=o.value_right,
                         sentiment=label)

            ro.add_opinion(no)

        return ro

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

        result = SingleClassEvalResult()
        for files_to_compare in files_to_compare_list:
            cmp_table, has_pos, has_neg = self.calc_a_file(files_to_compare, debug=debug)

            p, r = metrics.calc_prec_and_recall(cmp_table=cmp_table,
                                                label=self.__sentiment_label,
                                                opinions_exist=True,
                                                how_original_column=self.C_ORIG,
                                                how_results_column=self.C_RES,
                                                comparison_column=self.C_CMP)

            result.add_document_results(doc_id=files_to_compare.index,
                                        cmp_table=DocumentCompareTable(cmp_table),
                                        recall=r, prec=p)

        result.calculate()
        return result