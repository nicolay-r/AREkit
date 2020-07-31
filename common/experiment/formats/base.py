import logging
from os import path

from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.evaluation.results.base import BaseEvalResult
from arekit.common.experiment.data_io import DataIO
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.experiment.neutral.annot.three_scale import ThreeScaleNeutralAnnotator
from arekit.common.experiment.neutral.annot.two_scale import TwoScaleNeutralAnnotator
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler
from arekit.common.experiment.utils import get_path_of_subfolder_in_experiments_dir
from arekit.common.news.parsed.collection import ParsedNewsCollection

logger = logging.getLogger(__name__)


class BaseExperiment(object):

    def __init__(self, data_io, opin_operation, doc_operations, prepare_model_root):
        assert(isinstance(data_io, DataIO))
        assert(isinstance(prepare_model_root, bool))
        assert(isinstance(opin_operation, OpinionOperations))
        assert(isinstance(doc_operations, DocumentOperations))

        self.__opin_operations = opin_operation
        self.__doc_operations = doc_operations

        self.__data_io = data_io

        if prepare_model_root:
            self.DataIO.prepare_model_root()

        self.__neutral_annot = self.__init_annotator()

        # Setup DataIO
        self.__data_io.Callback.set_log_dir(path.join(self.DataIO.get_model_root(), u"log/"))

        # Initializing annotator
        logger.info("Initializing neutral annotator ...")
        self.__neutral_annot.initialize(data_io=data_io,
                                        opin_ops=self.OpinionOperations,
                                        doc_ops=self.DocumentOperations)

        # Setup model root
        model_root = self.DataIO.get_model_root()
        logger.info("Setup model root: {}".format(model_root))
        self.__data_io.ModelIO.set_model_root(value=model_root)

    # region Properties

    @property
    def Name(self):
        raise NotImplementedError()

    @property
    def DataIO(self):
        return self.__data_io

    @property
    def NeutralAnnotator(self):
        return self.__neutral_annot

    @property
    def OpinionOperations(self):
        return self.__opin_operations

    @property
    def DocumentOperations(self):
        return self.__doc_operations

    # endregion

    def get_input_samples_dir(self):
        is_fixed = self.__data_io.CVFoldingAlgorithm.CVCount == 1
        e_name = u"{name}_{mode}_{scale}l".format(name=self.Name,
                                                  mode=u"fixed" if is_fixed else u"cv",
                                                  scale=self.DataIO.LabelsScaler.LabelsCount)

        return get_path_of_subfolder_in_experiments_dir(subfolder_name=e_name,
                                                        experiments_dir=self.DataIO.get_experiments_dir())

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

        # Compose cmp pairs iterator.
        cmp_pairs_iter = self.__opin_operations.iter_opinion_collections_to_compare(
            data_type=data_type,
            doc_ids=self.__doc_operations.iter_news_indices(data_type=data_type),
            epoch_index=epoch_index)

        # getting evaluator.
        evaluator = self.__data_io.Evaluator
        assert(isinstance(evaluator, BaseEvaluator))

        # evaluate every document.
        result = evaluator.evaluate(cmp_pairs=cmp_pairs_iter)
        assert(isinstance(result, BaseEvalResult))

        # calculate results.
        result.calculate()

        return result

    # region private methods

    def __init_annotator(self):
        if isinstance(self.__data_io.LabelsScaler, TwoLabelScaler):
            return TwoScaleNeutralAnnotator()
        if isinstance(self.__data_io.LabelsScaler, ThreeLabelScaler):
            return ThreeScaleNeutralAnnotator()

    # endregion
