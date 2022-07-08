from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.labels.base import Label
from arekit.contrib.utils.evaluation.results.two_class_prf import TwoClassEvalPrecRecallF1Result


class TwoClassEvaluator(BaseEvaluator):

    def __init__(self, comparator, label1, label2, get_item_label_func):
        assert(isinstance(label1, Label))
        assert(isinstance(label2, Label))
        assert(callable(get_item_label_func))
        super(TwoClassEvaluator, self).__init__(comparator)

        self.__label1 = label1
        self.__label2 = label2
        self.__get_item_label_func = get_item_label_func

    def _create_eval_result(self):
        return TwoClassEvalPrecRecallF1Result(label1=self.__label1,
                                              label2=self.__label2,
                                              get_item_label_func=self.__get_item_label_func)
