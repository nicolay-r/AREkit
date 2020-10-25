import logging

import utils
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

            filepath = self._OpinOps.create_neutral_opinion_collection_filepath(
                doc_id=doc_id,
                data_type=data_type)

            if utils.check_file_already_existed(filepath=filepath, logger=logger):
                continue

            yield doc_id, filepath

    # endregion

    def iter_doc_ids_to_compare(self):
        doc_ids_iter = self.__iter_all_doc_ids()
        for doc_id in self._OpinOps.get_doc_ids_set_to_compare(doc_ids_iter):
            yield doc_id

    def initialize(self, opin_ops, doc_ops, synonyms):
        assert(isinstance(opin_ops, OpinionOperations))
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(synonyms, SynonymsCollection))
        # assert(data_io.CVFoldingAlgorithm.CVCount is not None)

        self.__doc_ops = doc_ops
        self.__opin_ops = opin_ops
        self.__synonyms = synonyms

    def create_collection(self, data_type, opinion_formatter):
        raise NotImplementedError()

    def _iter_docs(self, data_type):
        pairs = list(self.filter_non_created_doc_ids(data_type=data_type,
                                                     all_doc_ids=self.iter_doc_ids_to_compare()))

        if len(pairs) == 0:
            return pairs

        return progress_bar_iter(iterable=pairs,
                                 desc="Writing neutral-examples [{}]".format(data_type))


