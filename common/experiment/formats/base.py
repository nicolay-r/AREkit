import logging
from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.evaluation.results.base import BaseEvalResult
from arekit.common.experiment.data.base import DataIO
from arekit.common.experiment.data.training import TrainingData
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.experiment.neutral.annot.three_scale import ThreeScaleNeutralAnnotator
from arekit.common.experiment.neutral.annot.two_scale import TwoScaleNeutralAnnotator
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler
from arekit.common.news.parsed.collection import ParsedNewsCollection

logger = logging.getLogger(__name__)


class BaseExperiment(object):

    def __init__(self, data_io):
        assert(isinstance(data_io, DataIO))

        # Setup class fields
        self.__experiment_data = data_io
        self.__opin_operations = None
        self.__doc_operations = None
        # TODO. Maybe into data???
        self.__neutral_annot = self.__init_annotator()

    # region Properties

    @property
    def Name(self):
        raise NotImplementedError()

    @property
    def DataIO(self):
        return self.__experiment_data

    @property
    def OpinionOperations(self):
        return self.__opin_operations

    @property
    def DocumentOperations(self):
        return self.__doc_operations

    # TODO. Maybe into data???
    @property
    def NeutralAnnotator(self):
        return self.__neutral_annot

    # endregion

    def _set_opin_operations(self, value):
        assert(isinstance(value, OpinionOperations))
        self.__opin_operations = value

    def _set_doc_operations(self, value):
        assert(isinstance(value, DocumentOperations))
        self.__doc_operations = value

    def initialize_neutral_annotator(self):
        self.__neutral_annot.initialize(data_io=self.__experiment_data,
                                        opin_ops=self.__opin_operations,
                                        doc_ops=self.__doc_operations)

    def create_parsed_collection(self, data_type):
        assert(isinstance(data_type, DataType))

        parsed_news_it = self.DocumentOperations.iter_parsed_news(
            doc_inds=self.DocumentOperations.iter_news_indices(data_type))

        return ParsedNewsCollection(parsed_news_it)

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

        # Compose cmp pairs iterator.
        cmp_pairs_iter = self.__opin_operations.iter_opinion_collections_to_compare(
            data_type=data_type,
            doc_ids=self.__doc_operations.iter_news_indices(data_type=data_type),
            epoch_index=epoch_index)

        # getting evaluator.
        evaluator = self.__experiment_data.Evaluator
        assert(isinstance(evaluator, BaseEvaluator))

        # evaluate every document.
        result = evaluator.evaluate(cmp_pairs=cmp_pairs_iter)
        assert(isinstance(result, BaseEvalResult))

        # calculate results.
        result.calculate()

        return result

    # region private methods

    # TODO. Maybe into data???
    def __init_annotator(self):
        if isinstance(self.__experiment_data.LabelsScaler, TwoLabelScaler):
            return TwoScaleNeutralAnnotator()
        if isinstance(self.__experiment_data.LabelsScaler, ThreeLabelScaler):
            return ThreeScaleNeutralAnnotator()

    # TODO. Maybe into data???
    def get_annot_name(self):
        if isinstance(self.__neutral_annot, TwoScaleNeutralAnnotator):
            return u"neut_2_scale"
        if isinstance(self.__neutral_annot, ThreeScaleNeutralAnnotator):
            return u"neut_3_scale"

    # endregion
