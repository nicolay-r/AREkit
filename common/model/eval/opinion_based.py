import collections

from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.model.eval.base import BaseModelEvaluator


class OpinionBasedModelEvaluator(BaseModelEvaluator):

    def __init__(self, evaluator):
        assert(isinstance(evaluator, BaseEvaluator))
        self.__evaluator = evaluator

    # region abstract methods

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        raise NotImplementedError()

    def before_evaluation(self, data_type, doc_ids, epoch_index):
        pass

    # endregion

    # region public methods

    def evaluate(self, data_type, doc_ids, epoch_index):
        assert(isinstance(doc_ids, collections.Iterable))
        assert(isinstance(epoch_index, int))

        self.before_evaluation(data_type=data_type,
                               doc_ids=doc_ids,
                               epoch_index=epoch_index)

        opinions_cmp = self.iter_opinion_collections_to_compare(data_type=data_type,
                                                                doc_ids=doc_ids,
                                                                epoch_index=epoch_index)

        return self.__evaluator.evaluate(cmp_pairs=opinions_cmp)

    # endregion
