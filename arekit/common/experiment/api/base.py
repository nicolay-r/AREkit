import logging
from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.evaluation.results.base import BaseEvalResult
from arekit.common.evaluation.utils import OpinionCollectionsToCompareUtils
from arekit.common.experiment.api.ctx_base import ExperimentContext
from arekit.common.experiment.api.ctx_training import ExperimentTrainingContext
from arekit.common.experiment.api.enums import BaseDocumentTag
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.api.ops_opin import OpinionOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.utils import progress_bar_iter

logger = logging.getLogger(__name__)


class BaseExperiment(object):

    def __init__(self, exp_ctx, exp_io, opin_ops, doc_ops):
        assert(isinstance(exp_ctx, ExperimentContext))
        assert(isinstance(exp_io, BaseIOUtils))
        assert(isinstance(opin_ops, OpinionOperations))
        assert(isinstance(doc_ops, DocumentOperations))
        self.__exp_ctx = exp_ctx
        self.__exp_io = exp_io
        self.__opin_ops = opin_ops
        self.__doc_ops = doc_ops

    # region Properties

    @property
    def ExperimentContext(self):
        return self.__exp_ctx

    @property
    def ExperimentIO(self):
        """ Filepaths, related to experiment
        """
        return self.__exp_io

    @property
    def OpinionOperations(self):
        return self.__opin_ops

    @property
    def DocumentOperations(self):
        return self.__doc_ops

    # endregion

    def _init_log_flag(self, do_log):
        assert(isinstance(do_log, bool))
        self._do_log = do_log

    def log_info(self, message, forced=False):
        assert (isinstance(message, str))
        if not self._do_log and not forced:
            return
        logger.info(message)

    def evaluate(self, data_type, epoch_index):
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
        assert(isinstance(self.__exp_ctx, ExperimentTrainingContext))

        # Extracting all docs to cmp and those that is related to data_type.
        cmp_doc_ids_iter = self.__doc_ops.iter_tagget_doc_ids(BaseDocumentTag.Compare)
        doc_ids_iter = self.__doc_ops.iter_doc_ids(data_type=data_type)
        cmp_doc_ids_set = set(cmp_doc_ids_iter)

        # Compose cmp pairs iterator.
        cmp_pairs_iter = OpinionCollectionsToCompareUtils.iter_comparable_collections(
            doc_ids=[doc_id for doc_id in doc_ids_iter if doc_id in cmp_doc_ids_set],
            read_etalon_collection_func=lambda doc_id: self.__opin_ops.get_etalon_opinion_collection(
                doc_id=doc_id),
            read_result_collection_func=lambda doc_id: self.__opin_ops.get_result_opinion_collection(
                data_type=data_type,
                doc_id=doc_id,
                epoch_index=epoch_index))

        # getting evaluator.
        evaluator = self.__exp_ctx.Evaluator
        assert(isinstance(evaluator, BaseEvaluator))

        # evaluate every document.
        logged_cmp_pairs_it = progress_bar_iter(cmp_pairs_iter, desc="Evaluate", unit='pairs')
        result = evaluator.evaluate(cmp_pairs=logged_cmp_pairs_it)
        assert(isinstance(result, BaseEvalResult))

        # calculate results.
        result.calculate()

        return result
