from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.contrib.utils.evaluation.results.two_class import TwoClassEvalResult


class TwoClassEvaluator(BaseEvaluator):

    def _create_eval_result(self):
        return TwoClassEvalResult()
