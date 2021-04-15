from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.evaluation.results.two_class import TwoClassEvalResult


# TODO. This class suppose to be a part of the RuSentRel experiment.
# (It refers to sentiment labels which should be moved out of core)
class TwoClassEvaluator(BaseEvaluator):

    def __init__(self, eval_mode=EvaluationModes.Extraction):
        super(TwoClassEvaluator, self).__init__(eval_mode=eval_mode)

    def _create_eval_result(self):
        return TwoClassEvalResult()
