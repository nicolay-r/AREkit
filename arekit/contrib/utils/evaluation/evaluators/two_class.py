from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.labels.base import Label
from arekit.contrib.utils.evaluation.results.two_class import TwoClassEvalResult


class TwoClassEvaluator(BaseEvaluator):

    def __init__(self, comparator, label1, label2):
        assert(isinstance(label1, Label))
        assert(isinstance(label2, Label))
        super(TwoClassEvaluator, self).__init__(comparator)

        self.__label1 = label1
        self.__label2 = label2

    def _create_eval_result(self):
        return TwoClassEvalResult(label1=self.__label1, label2=self.__label2)
