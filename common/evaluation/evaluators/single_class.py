import collections

from arekit.common.labels.base import Label
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.opinions.base import Opinion
from arekit.common.evaluation.cmp_opinions import OpinionCollectionsToCompare
from arekit.common.evaluation.evaluators import metrics
from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.evaluation.results.single_class import SingleClassEvalResult


class SentimentLabel(Label):

    def __init__(self):
        pass


class SingleClassEvaluator(BaseEvaluator):

    def __init__(self, create_synonyms_collection_func):
        assert(callable(create_synonyms_collection_func))
        super(SingleClassEvaluator, self).__init__()
        self.__create_synonyms_collection_func = create_synonyms_collection_func
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

        new_collection = self.__create_synonyms_collection_func()

        for opinion in opinions:
            assert(isinstance(opinion, Opinion))
            new_opinion = Opinion(source_value=opinion.SourceValue,
                                  target_value=opinion.TargetValue,
                                  sentiment=label)

            new_collection.add_opinion(new_opinion)

        return new_collection

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
