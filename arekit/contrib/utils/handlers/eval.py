from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.evaluation.result import BaseEvalResult
from arekit.common.experiment.api.enums import BaseDocumentTag
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.handler import ExperimentIterationHandler
from arekit.common.utils import progress_bar_iter
from arekit.contrib.utils.evaluation.iterators import DataPairsIterators


class EvalIterationHandler(ExperimentIterationHandler):
    """ TODO: #355 affected.
        Этот класс устарел ввиду зависимостей на Opinion, в то время как
        нас в оценке результатов могут, например, интересовать TextOpinion.
        Поэтому здесь нужно избавиться от opin_ops, а также возможно что
        от класса в целом, так как предполагаются отдельные реализации
        различных оценок в виде отдельных функций.
    """

    # TODO. #366 add cmp_doc_ops.
    def __init__(self, data_type, doc_ops, epoch_indices, evaluator,
                 get_test_doc_collection_func, get_etalon_doc_collection_func):
        """ get_doc_collection_func: func
                (doc_id) -> collection (Any type)
        """
        assert(isinstance(data_type, DataType))
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(epoch_indices, list))
        assert(isinstance(evaluator, BaseEvaluator))
        assert(callable(get_test_doc_collection_func))
        assert(callable(get_etalon_doc_collection_func))

        self.__data_type = data_type
        self.__doc_ops = doc_ops
        # TODO. #366 add cmp_doc_ops.
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

        # Extracting all docs to cmp and those that is related to data_type.
        # TODO. #366. use iter_doc_ids. 2) rename __doc_ops to __cmp_doc_ops.
        cmp_doc_ids_iter = self.__doc_ops.iter_tagget_doc_ids(BaseDocumentTag.Compare)
        doc_ids_iter = self.__doc_ops.iter_doc_ids(data_type=data_type)
        cmp_doc_ids_set = set(cmp_doc_ids_iter)

        # Compose cmp pairs iterator.
        cmp_pairs_iter = DataPairsIterators.iter_func_based_collections(
            doc_ids=[doc_id for doc_id in doc_ids_iter if doc_id in cmp_doc_ids_set],
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
