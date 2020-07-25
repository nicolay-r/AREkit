import logging
import utils
from arekit.common.experiment.data_io import DataIO
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseNeutralAnnotator(object):
    """
    Performs neutral annotation for different data_type.
    """

    def __init__(self, annot_name):
        assert(isinstance(annot_name, unicode))

        logger.info("Init annotator: [{}]".format(annot_name))
        self.__annot_name = annot_name
        self.__opin_ops = None
        self.__doc_ops = None
        self.__data_io = None

    @property
    def AnnotatorName(self):
        return self.__annot_name

    @property
    def _OpinOps(self):
        assert(isinstance(self.__opin_ops, OpinionOperations))
        return self.__opin_ops

    @property
    def _DocOps(self):
        assert(isinstance(self.__doc_ops, DocumentOperations))
        return self.__doc_ops

    @property
    def _DataIO(self):
        assert(isinstance(self.__data_io, DataIO))
        return self.__data_io

    # region private methods

    def __iter_all_doc_ids(self):
        for data_type in self._DocOps.iter_suppoted_data_types():
            for doc_id in self._DocOps.iter_news_indices(data_type):
                yield doc_id

    def filter_non_created_doc_ids(self, all_doc_ids, data_type):

        for doc_id in all_doc_ids:

            filepath = self._OpinOps.create_neutral_opinion_collection_filepath(
                doc_id=doc_id,
                data_type=data_type)

            if utils.check_file_already_existed(filepath=filepath, logger=logger):
                continue

            yield doc_id, filepath

    # endregion

    def iter_doc_ids_to_compare(self):
        doc_ids_iter = self.__iter_all_doc_ids()
        for doc_id in self._OpinOps.iter_doc_ids_to_compare(doc_ids_iter):
            yield doc_id

    def initialize(self, data_io, opin_ops, doc_ops):
        assert(isinstance(data_io, DataIO))
        assert(isinstance(opin_ops, OpinionOperations))
        assert(isinstance(doc_ops, DocumentOperations))
        assert(data_io.CVFoldingAlgorithm.CVCount is not None)

        self.__doc_ops = doc_ops
        self.__opin_ops = opin_ops
        self.__data_io = data_io

    def create_collection(self, data_type):
        raise NotImplementedError()


