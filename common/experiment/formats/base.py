import logging
from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.evaluation.results.base import BaseEvalResult
from arekit.common.evaluation.utils import OpinionCollectionsToCompareUtils
from arekit.common.experiment.data.base import DataIO
from arekit.common.experiment.data.training import TrainingData
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.experiment.io_utils import BaseIOUtils
from arekit.common.news.parsed.collection import ParsedNewsCollection

logger = logging.getLogger(__name__)


class BaseExperiment(object):

    def __init__(self, data_io, experiment_io):
        assert(isinstance(data_io, DataIO))
        assert(issubclass(experiment_io, BaseIOUtils))
        self.__experiment_data = data_io
        self.__experiment_io = experiment_io(self)
        self.__opin_operations = None
        self.__doc_operations = None

    # region Properties

    @property
    def Name(self):
        raise NotImplementedError()

    @property
    def DataIO(self):
        """ TODO. Should be renamed
            Related to extra resources, utlized in experiment organization.
        """
        return self.__experiment_data

    @property
    def ExperimentIO(self):
        """ Filepaths, related to experiment
        """
        return self.__experiment_io

    @property
    def OpinionOperations(self):
        return self.__opin_operations

    @property
    def DocumentOperations(self):
        return self.__doc_operations

    # endregion

    # region protected method

    def _set_opin_operations(self, value):
        assert(isinstance(value, OpinionOperations))
        self.__opin_operations = value

    def _set_doc_operations(self, value):
        assert(isinstance(value, DocumentOperations))
        self.__doc_operations = value

    # endregion

    def initialize_neutral_annotator(self):
        self.__experiment_data.NeutralAnnotator.initialize(
            synonyms=self.__experiment_data.SynonymsCollection,
            opin_ops=self.__opin_operations,
            doc_ops=self.__doc_operations)

    def create_parsed_collection(self, data_type):
        assert(isinstance(data_type, DataType))

        parsed_news_it = self.DocumentOperations.iter_parsed_news(
            doc_inds=self.DocumentOperations.iter_news_indices(data_type))

        return ParsedNewsCollection(parsed_news_it, notify=True)

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
        assert(isinstance(self.__experiment_data, TrainingData))

        # Extracting all docs to cmp and those that is related to data_type.
        cmp_doc_ids = self.__doc_operations.iter_doc_ids_to_compare()
        doc_ids = self.__doc_operations.iter_news_indices(data_type=data_type)

        # Compose cmp pairs iterator.
        cmp_pairs_iter = OpinionCollectionsToCompareUtils.iter_comparable_collections(
            doc_ids=filter(lambda doc_id: doc_id in cmp_doc_ids, doc_ids),
            read_etalon_collection_func=lambda doc_id: self.__opin_operations.read_etalon_opinion_collection(
                doc_id=doc_id),
            read_result_collection_func=lambda doc_id: self.__opin_operations.read_result_opinion_collection(
                data_type=data_type,
                doc_id=doc_id,
                epoch_index=epoch_index))

        # getting evaluator.
        evaluator = self.__experiment_data.Evaluator
        assert(isinstance(evaluator, BaseEvaluator))

        # evaluate every document.
        result = evaluator.evaluate(cmp_pairs=cmp_pairs_iter)
        assert(isinstance(result, BaseEvalResult))

        # calculate results.
        result.calculate()

        return result
