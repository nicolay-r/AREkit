import logging
from os import path

from arekit.common.experiment.data_io import DataIO
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.experiment.neutral.annot.three_scale import ThreeScaleNeutralAnnotator
from arekit.common.experiment.neutral.annot.two_scale import TwoScaleNeutralAnnotator
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler
from arekit.common.parsed_news.collection import ParsedNewsCollection

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
        # TODO. Move into data_io
        self.__data_io.Callback.set_log_dir(log_dir=path.join(self.DataIO.get_model_root(), u"log/"))

        self.__neutral_annot.initialize(data_io=data_io,
                                        opin_ops=self.OpinionOperations,
                                        doc_ops=self.DocumentOperations)

        self.__data_io.ModelIO.set_model_root(value=self.DataIO.get_model_root())

    # region Properties

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

    def create_parsed_collection(self, data_type, parse_frame_variants=True):
        assert(isinstance(data_type, DataType))
        assert(isinstance(parse_frame_variants, bool))

        parsed_news_it = self.DocumentOperations.iter_parsed_news(
            doc_inds=self.DocumentOperations.iter_news_indices(data_type),
            frame_variant_collection=self.DataIO.FrameVariantCollection if parse_frame_variants else None)

        return ParsedNewsCollection(parsed_news_it)

    # region private methods

    def __init_annotator(self):
        if isinstance(self.__data_io.LabelsScaler, TwoLabelScaler):
            return TwoScaleNeutralAnnotator()
        if isinstance(self.__data_io.LabelsScaler, ThreeLabelScaler):
            return ThreeScaleNeutralAnnotator()

    # endregion
