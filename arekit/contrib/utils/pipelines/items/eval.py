from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.evaluation.result import BaseEvalResult
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.utils import progress_bar_iter
from arekit.contrib.utils.evaluation.iterators import DataPairsIterators
from arekit.contrib.utils.model_io.utils import folding_iter_states


class EvaluationPipelineItem(BasePipelineItem):

    def __init__(self, data_type, cmp_data_folding, evaluator,
                 get_test_doc_collection_func, get_etalon_doc_collection_func):
        """ get_doc_collection_func: func
                (doc_id) -> collection (Any type)
        """
        assert(isinstance(data_type, DataType))
        assert(isinstance(cmp_data_folding, NoFolding))
        assert(isinstance(evaluator, BaseEvaluator))
        assert(callable(get_test_doc_collection_func))
        assert(callable(get_etalon_doc_collection_func))

        self.__data_type = data_type
        self.__cmp_data_folding = cmp_data_folding
        self.__evaluator = evaluator
        self.__get_test_doc_collection_func = get_test_doc_collection_func
        self.__get_etalon_doc_collection_func = get_etalon_doc_collection_func

    def __evaluate(self, data_type):
        """
        Perform experiment evaluation (related model) of a certain
        `data_type` at certain `epoch_index`

        data_type: DataType
            used as data source (for document ids)

        NOTE: assumes that results already written and converted in doc-level opinions.
        """
        assert(isinstance(data_type, DataType))

        _, cmp_doc_ids = next(iter(self.__cmp_data_folding.fold_doc_ids_set()))

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
        """
        results = []

        for _ in folding_iter_states(self.__cmp_data_folding):
            results.append(self.__evaluate(data_type=self.__data_type))

        return results
