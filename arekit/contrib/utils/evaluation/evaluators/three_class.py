from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.labels.base import Label
from arekit.contrib.utils.evaluation.results.three_class_prf import ThreeClassPrecRecallF1EvalResult


class ThreeClassEvaluator(BaseEvaluator):

    def __init__(self, comparator, label1, label2, label3, get_item_label_func):
        assert(isinstance(label1, Label))
        assert(isinstance(label2, Label))
        assert(isinstance(label3, Label))
        assert(callable(get_item_label_func))
        super(ThreeClassEvaluator, self).__init__(comparator)

        self.__label1 = label1
        self.__label2 = label2
        self.__label3 = label3
        self.__get_item_label_func = get_item_label_func

    def _create_eval_result(self):
        return ThreeClassPrecRecallF1EvalResult(label1=self.__label1,
                                                label2=self.__label2,
                                                label3=self.__label3,
                                                get_item_label_func=self.__get_item_label_func)
