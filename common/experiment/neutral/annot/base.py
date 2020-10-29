import logging

from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.synonyms import SynonymsCollection
from arekit.common.utils import progress_bar_iter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseNeutralAnnotator(object):
    """
    Performs neutral annotation for different data_type.
    """

    def __init__(self):
        logger.info("Init annotator: [{}]".format(self.__class__))
        self.__opin_ops = None
        self.__doc_ops = None
        self.__synonyms = None

    @property
    def Name(self):
        raise NotImplementedError()

    @property
    def _OpinOps(self):
        assert(isinstance(self.__opin_ops, OpinionOperations))
        return self.__opin_ops

    @property
    def _DocOps(self):
        assert(isinstance(self.__doc_ops, DocumentOperations))
        return self.__doc_ops

    @property
    def _SynonymsCollection(self):
        return self.__synonyms

    # region private methods

    def __iter_all_doc_ids(self):
        for data_type in self._DocOps.iter_supported_data_types():
            for doc_id in self._DocOps.iter_news_indices(data_type):
                yield doc_id

    def filter_non_created_doc_ids(self, all_doc_ids, data_type):
        for doc_id in all_doc_ids:
            if self._OpinOps.try_read_neutral_opinion_collection(doc_id=doc_id, data_type=data_type) is None:
                yield doc_id

    # endregion

    def _iter_doc_its_to_annotate(self):
        return filter(lambda doc_id: doc_id in self._DocOps.get_doc_ids_set_to_neutrally_annotate(),
                      self.__iter_all_doc_ids())

    def initialize(self, opin_ops, doc_ops, synonyms):
        assert(isinstance(opin_ops, OpinionOperations))
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(synonyms, SynonymsCollection))
        # assert(data_io.CVFoldingAlgorithm.CVCount is not None)

        self.__doc_ops = doc_ops
        self.__opin_ops = opin_ops
        self.__synonyms = synonyms

    def create_collection(self, data_type):
        raise NotImplementedError()

    def _iter_docs(self, data_type):
        doc_ids_iter = self.filter_non_created_doc_ids(
            data_type=data_type,
            all_doc_ids=self._iter_doc_its_to_annotate())

        doc_ids = list(doc_ids_iter)

        if len(doc_ids) == 0:
            return doc_ids

        return progress_bar_iter(iterable=doc_ids,
                                 desc="Writing neutral-examples [{}]".format(data_type))


