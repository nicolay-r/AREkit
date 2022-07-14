import collections

from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.evaluation.result import BaseEvalResult
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.utils import progress_bar_iter
from arekit.contrib.utils.evaluation.iterators import DataPairsIterators
from arekit.contrib.utils.utils_folding import folding_iter_states


class EvaluationPipelineItem(BasePipelineItem):

    def __init__(self, evaluator, get_test_doc_collection_func, get_etalon_doc_collection_func):
        """ get_doc_collection_func: func
                (doc_id) -> collection (Any type)
        """
        assert(isinstance(evaluator, BaseEvaluator))
        assert(callable(get_test_doc_collection_func))
        assert(callable(get_etalon_doc_collection_func))

        self.__evaluator = evaluator
        self.__get_test_doc_collection_func = get_test_doc_collection_func
        self.__get_etalon_doc_collection_func = get_etalon_doc_collection_func

    def __evaluate(self, cmp_doc_ids):
        assert(isinstance(cmp_doc_ids, collections.Iterable))

        # Compose cmp pairs iterator.
        cmp_pairs_iter = DataPairsIterators.iter_func_based_collections(
            doc_ids=cmp_doc_ids,
            read_etalon_collection_func=lambda doc_id: self.__get_test_doc_collection_func(doc_id),
            read_test_collection_func=lambda doc_id: self.__get_etalon_doc_collection_func(doc_id))

        # evaluate every document.
        logged_cmp_pairs_it = progress_bar_iter(cmp_pairs_iter, desc="Evaluate", unit='pairs')
        result = self.__evaluator.evaluate(cmp_pairs=logged_cmp_pairs_it)
        assert(isinstance(result, BaseEvalResult))

        # calculate results.
        result.calculate()

        return result

    def apply_core(self, input_data, pipeline_ctx):
        """ Provide results per every iteration state of the cmp_data_folding

        data_type: DataType
            used as data source (for document ids)
        """
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert("data_type" in pipeline_ctx)
        assert("cmp_data_folding" in pipeline_ctx)

        results = []
        data_type = pipeline_ctx.provide("data_type")
        cmp_data_folding = pipeline_ctx.provide("cmp_data_folding")
        for _ in folding_iter_states(cmp_data_folding):
            _, cmp_doc_ids = cmp_data_folding.fold_doc_ids_set()[data_type]
            results.append(self.__evaluate(cmp_doc_ids))

        return results
