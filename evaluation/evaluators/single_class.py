import collections

from core.evaluation.cmp_opinions import OpinionCollectionsToCompare
from core.evaluation.evaluators.base import BaseEvaluator
from core.evaluation.labels import Label
from core.evaluation.results.single_class import SingleClassEvalResult
from core.common.opinions.collection import OpinionCollection
from core.common.opinions.opinion import Opinion
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

    def calc_a_file(self, cmp_pair):
        assert(isinstance(cmp_pair, OpinionCollectionsToCompare))

        test_opins = self.__clone_with_different_label(opinions=cmp_pair.TestOpinionCollection,
                                                       label=self.__sentiment_label)
        etalon_opins = self.__clone_with_different_label(opinions=cmp_pair.EtalonOpinionCollection,
                                                         label=self.__sentiment_label)

        cmp_table = self.calc_difference(etalon_opins, test_opins)

        return cmp_table

    def __clone_with_different_label(self, opinions, label):
        assert(isinstance(opinions, OpinionCollection))
        assert(isinstance(label, Label))

        ro = OpinionCollection(opinions=[],
                               synonyms=self.Synonyms)

        for o in opinions:
            assert(isinstance(o, Opinion))
            no = Opinion(source_value=o.SourceValue,
                         target_value=o.TargetValue,
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

    def evaluate(self, cmp_pairs):
        assert(isinstance(cmp_pairs, collections.Iterable))

        result = SingleClassEvalResult()
        for cmp_pair in cmp_pairs:
            cmp_table, has_pos, has_neg = self.calc_a_file(cmp_pair)

            p, r = metrics.calc_prec_and_recall(cmp_table=cmp_table,
                                                label=self.__sentiment_label,
                                                opinions_exist=True)

            result.add_document_results(doc_id=cmp_pair.DocumentID,
                                        cmp_table=cmp_table,
                                        recall=r, prec=p)

        result.calculate()
        return result
