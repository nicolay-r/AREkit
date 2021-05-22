import logging

from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.news.parsed.base import ParsedNews
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

    @property
    def LabelsCount(self):
        raise NotImplementedError()

    # region Properties

    @property
    def _OpinOps(self):
        assert(isinstance(self.__opin_ops, OpinionOperations))
        return self.__opin_ops

    @property
    def _DocOps(self):
        assert(isinstance(self.__doc_ops, DocumentOperations))
        return self.__doc_ops

    # endregion

    # region private methods

    def __iter_neutral_collections(self, data_type, filter_func):
        docs_to_annot_list = filter(filter_func,
                                    self._DocOps.iter_doc_ids_to_neutrally_annotate())

        if len(docs_to_annot_list) == 0:
            logger.info("[{}]: OK!".format(data_type))
            return

        logged_parsed_news_iter = progress_bar_iter(
            iterable=self._DocOps.iter_parsed_news(docs_to_annot_list),
            desc="Creating neutral-examples [{}]".format(data_type))

        for parsed_news in logged_parsed_news_iter:
            assert(isinstance(parsed_news, ParsedNews))
            yield parsed_news.RelatedNewsID, \
                  self._create_collection_core(parsed_news=parsed_news, data_type=data_type)

    # endregion

    def _create_collection_core(self, parsed_news, data_type):
        raise NotImplementedError

    # region public methods

    def initialize(self, opin_ops, doc_ops):
        assert(isinstance(opin_ops, OpinionOperations))
        assert(isinstance(doc_ops, DocumentOperations))
        self.__doc_ops = doc_ops
        self.__opin_ops = opin_ops

    def serialize_missed_collections(self, data_type):

        filter_func = lambda doc_id: self._OpinOps.try_read_neutrally_annotated_opinion_collection(
            doc_id=doc_id, data_type=data_type) is None

        for doc_id, collection in self.__iter_neutral_collections(data_type, filter_func):
            self._OpinOps.save_neutrally_annotated_opinion_collection(collection=collection,
                                                                      doc_id=doc_id,
                                                                      data_type=data_type)

    # endregion