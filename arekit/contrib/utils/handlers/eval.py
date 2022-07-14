from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.evaluation.result import BaseEvalResult
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.handler import ExperimentIterationHandler
from arekit.common.folding.nofold import NoFolding
from arekit.common.utils import progress_bar_iter
from arekit.contrib.utils.evaluation.iterators import DataPairsIterators


class EvalIterationHandler(ExperimentIterationHandler):

    def __init__(self, data_type, cmp_data_folding, epoch_indices, evaluator,
                 get_test_doc_collection_func, get_etalon_doc_collection_func):
        """ get_doc_collection_func: func
                (doc_id) -> collection (Any type)
        """
        assert(isinstance(data_type, DataType))
        assert(isinstance(cmp_data_folding, NoFolding))
        assert(isinstance(epoch_indices, list))
        assert(isinstance(evaluator, BaseEvaluator))
        assert(callable(get_test_doc_collection_func))
        assert(callable(get_etalon_doc_collection_func))

        self.__data_type = data_type
        self.__cmp_data_folding = cmp_data_folding
        self.__epoch_indices = epoch_indices
        self.__evaluator = evaluator
        self.__get_test_doc_collection_func = get_test_doc_collection_func
        self.__get_etalon_doc_collection_func = get_etalon_doc_collection_func

    def __evaluate(self, data_type, epoch_index):
        """
        Perform experiment evaluation (related model) of a certain
        `data_type` at certain `epoch_index`

        data_type: DataType
            used as data source (for document ids)
        epoch_index: int or None

        NOTE: assumes that results already written and converted in doc-level opinions.
        """
        assert(isinstance(data_type, DataType))
        assert(isinstance(epoch_index, int))

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

    def on_iteration(self, iter_index):
        for epoch in self.__epoch_indices:
            self.__evaluate(data_type=self.__data_type, epoch_index=epoch)
