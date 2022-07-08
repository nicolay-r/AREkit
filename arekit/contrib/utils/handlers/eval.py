from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.evaluation.result import BaseEvalResult
from arekit.common.experiment.api.enums import BaseDocumentTag
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.api.ops_opin import OpinionOperations
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

    def __init__(self, data_type, doc_ops, opin_ops, epoch_indices, evaluator):
        assert(isinstance(data_type, DataType))
        assert(isinstance(doc_ops, DocumentOperations))
        # TODO. #355 related. OpinionOperations limit this onto `Opinion` type only.
        assert(isinstance(opin_ops, OpinionOperations))
        assert(isinstance(epoch_indices, list))
        assert(isinstance(evaluator, BaseEvaluator))

        self.__data_type = data_type
        self.__doc_ops = doc_ops
        self.__opin_ops = opin_ops
        self.__epoch_indices = epoch_indices
        self.__evaluator = evaluator

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
        cmp_doc_ids_iter = self.__doc_ops.iter_tagget_doc_ids(BaseDocumentTag.Compare)
        doc_ids_iter = self.__doc_ops.iter_doc_ids(data_type=data_type)
        cmp_doc_ids_set = set(cmp_doc_ids_iter)

        # Compose cmp pairs iterator.
        cmp_pairs_iter = DataPairsIterators.iter_func_based_collections(
            doc_ids=[doc_id for doc_id in doc_ids_iter if doc_id in cmp_doc_ids_set],
            # TODO. #355 related. OpinionOperations limit this onto `Opinion` type only.
            read_etalon_collection_func=lambda doc_id: self.__opin_ops.get_etalon_opinion_collection(
                doc_id=doc_id),
            # TODO. #355 related. OpinionOperations limit this onto `Opinion` type only.
            read_test_collection_func=lambda doc_id: self.__opin_ops.get_result_opinion_collection(
                data_type=data_type,
                doc_id=doc_id,
                epoch_index=epoch_index))

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
