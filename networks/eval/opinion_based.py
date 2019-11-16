import collections

from core.evaluation.evaluators.base import BaseEvaluator
from core.networks.eval.base import EvaluationHelper
from core.networks.network_io import NetworkIO


class OpinionBasedEvaluationHelper(EvaluationHelper):

    def __init__(self, evaluator):
        assert(isinstance(evaluator, BaseEvaluator))
        self.__evaluator = evaluator

    def evaluate_model(self, data_type, io, doc_ids, epoch_index):
        assert(isinstance(io, NetworkIO))
        assert(isinstance(doc_ids, collections.Iterable))
        assert(isinstance(epoch_index, int))

        opinions_cmp = io.iter_opinion_collections_to_compare(data_type=data_type,
                                                              doc_ids=doc_ids,
                                                              epoch_index=epoch_index)

        return self.__evaluator.evaluate(cmp_pairs=opinions_cmp)
