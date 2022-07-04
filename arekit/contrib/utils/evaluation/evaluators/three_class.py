from arekit.common.evaluation.comparators.opinions import OpinionBasedComparator
from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.evaluators.utils import check_is_supported
from arekit.common.labels.base import Label
from arekit.common.opinions.collection import OpinionCollection
from arekit.contrib.utils.evaluation.results.three_class import ThreeClassEvalResult


class ThreeClassOpinionEvaluator(BaseEvaluator):
    """ This evaluator is limitied and applied for OpinionCollections.
    """

    def __init__(self, label1, label2, no_label):
        """ Since we consider that NEU labels are not a part of the result data.
            Therefore if something missed in results then it suppose to be correct classified.
        """
        assert(isinstance(label1, Label))
        assert(isinstance(label2, Label))
        assert(isinstance(no_label, Label))
        comparator = OpinionBasedComparator(EvaluationModes.Extraction)

        self.__label1 = label1
        self.__label2 = label2
        self.__no_label = no_label

        super(ThreeClassOpinionEvaluator, self).__init__(comparator)

    def _calc_diff(self, etalon_data, test_data, is_label_supported):
        assert(isinstance(etalon_data, OpinionCollection))
        assert(isinstance(test_data, OpinionCollection))

        # Composing test opinion collection
        # by combining already existed with
        # the neutrally labeled opinions.
        test_opins_expanded = test_data.copy()

        for opinion in etalon_data:
            # We keep only those opinions that were not
            # presented in test and has neutral label

            check_is_supported(label=opinion.Sentiment, is_label_supported=is_label_supported)

            if not test_opins_expanded.has_synonymous_opinion(opinion) and opinion.Sentiment == self.__no_label:
                test_opins_expanded.add_opinion(opinion)

        return super(ThreeClassOpinionEvaluator, self)._calc_diff(etalon_data=etalon_data,
                                                                  test_data=test_opins_expanded,
                                                                  is_label_supported=is_label_supported)

    def _create_eval_result(self):
        return ThreeClassEvalResult(label1=self.__label1,
                                    label2=self.__label2,
                                    no_label=self.__no_label,
                                    get_item_label_func=lambda opinion: opinion.Sentiment)
