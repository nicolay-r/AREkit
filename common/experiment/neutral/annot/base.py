import logging

from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.synonyms import SynonymsCollection
from arekit.common.utils import progress_bar_iter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseNeutralAnnotator(object):
    """
    Performs neutral annotation for different data_type.
    """

    def __init__(self, labels_fmt):
        assert(isinstance(labels_fmt, StringLabelsFormatter))

        logger.info("Init annotator: [{}]".format(self.__class__))

        self.__labels_fmt = labels_fmt
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

    def __iter_docs(self, data_type, filter_func):
        doc_ids_iter = filter(filter_func,
                              self._DocOps.iter_doc_ids_to_neutrally_annotate())

        doc_ids = list(doc_ids_iter)

        if len(doc_ids) == 0:
            return doc_ids

        return progress_bar_iter(iterable=doc_ids,
                                 desc="Writing neutral-examples [{}]".format(data_type))

    def __iter_neutral_collections(self, data_type, filter_func):
        for doc_id in self.__iter_docs(data_type, filter_func=filter_func):
            yield doc_id, self._create_collection_core(doc_id=doc_id, data_type=data_type)

    # endregion

    def _create_collection_core(self, doc_id, data_type):
        raise NotImplementedError

    # region public methods

    def initialize(self, opin_ops, doc_ops, synonyms):
        assert(isinstance(opin_ops, OpinionOperations))
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(synonyms, SynonymsCollection))

        self.__doc_ops = doc_ops
        self.__opin_ops = opin_ops
        self.__synonyms = synonyms

    def serialize_missed_collections(self, data_type):

        filter_func = lambda doc_id: self._OpinOps.try_read_neutral_opinion_collection(
            doc_id=doc_id, data_type=data_type) is None

        for doc_id, collection in self.__iter_neutral_collections(data_type, filter_func):
            self._OpinOps.save_neutral_opinion_collection(collection=collection,
                                                          labels_fmt=self.__labels_fmt,
                                                          doc_id=doc_id,
                                                          data_type=data_type)

    # endregion