from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.results.three_class import ThreeClassEvalResult
from arekit.common.experiment.data_type import DataType
from arekit.common.opinions.collection import OpinionCollection


class ThreeClassEvaluator(BaseEvaluator):

    def __init__(self, data_type):
        """ Evaluation additionally depends on utilized data_type.
            Since we consider that NEU labels are not a part of the result data.
            Therefore if something missed in results then it suppose to be correct classified.
        """
        assert(isinstance(data_type, DataType))
        super(ThreeClassEvaluator, self).__init__(eval_mode=EvaluationModes.Extraction)
        self.__data_type = data_type

    def _calc_diff(self, etalon_opins, test_opins):
        assert(isinstance(etalon_opins, OpinionCollection))
        assert(isinstance(test_opins, OpinionCollection))

        neut_label = ThreeClassEvalResult.create_neutral_label()

        # Composing test opinion collection
        # by combining already existed with
        # the neutrally labeled opinions.
        test_opins_expanded = test_opins.copy()

        for opinion in etalon_opins:
            # We keep only those opinions that were not
            # presented in test and has neutral label
            if not test_opins_expanded.has_synonymous_opinion(opinion) and opinion.Sentiment == neut_label:
                test_opins_expanded.add_opinion(opinion)

        return super(ThreeClassEvaluator, self)._calc_diff(etalon_opins=etalon_opins,
                                                           test_opins=test_opins_expanded)

    def _create_eval_result(self):
        return ThreeClassEvalResult()
