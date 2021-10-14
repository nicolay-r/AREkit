import logging

from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.api.ops_opin import OpinionOperations
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.utils import progress_bar_iter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseAnnotator(object):
    """
    Performs annotation for a particular data_type
    using OpinOps and DocOps API.
    """

    def __init__(self):
        logger.info("Init annotator: [{}]".format(self.__class__))

    @property
    def LabelsCount(self):
        raise NotImplementedError()

    # region private methods

    def __iter_annotated_collections(self, data_type, filter_func, doc_ops, opin_ops):
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(opin_ops, OpinionOperations))

        docs_to_annot_list = list(filter(filter_func,
                                  doc_ops.iter_doc_ids_to_annotate()))

        if len(docs_to_annot_list) == 0:
            logger.info("[{}]: Nothing to annotate".format(data_type))
            return

        logged_parsed_news_iter = progress_bar_iter(
            iterable=doc_ops.iter_parsed_docs(docs_to_annot_list),
            desc="Annotating parsed news [{}]".format(data_type))

        for parsed_news in logged_parsed_news_iter:
            assert(isinstance(parsed_news, ParsedNews))
            yield parsed_news.RelatedNewsID, \
                  self._annot_collection_core(parsed_news=parsed_news, data_type=data_type,
                                              doc_ops=doc_ops, opin_ops=opin_ops)

    # endregion

    def _annot_collection_core(self, parsed_news, data_type, doc_ops, opin_ops):
        raise NotImplementedError

    # region public methods

    def serialize_missed_collections(self, data_type, doc_ops, opin_ops):
        assert(isinstance(opin_ops, OpinionOperations))

        filter_func = lambda doc_id: opin_ops.try_read_annotated_opinion_collection(
            doc_id=doc_id, data_type=data_type) is None

        annot_it = self.__iter_annotated_collections(
            data_type,
            filter_func,
            doc_ops=doc_ops,
            opin_ops=opin_ops)

        for doc_id, collection in annot_it:
            # TODO. Save here is weird.
            opin_ops.save_annotated_opinion_collection(collection=collection,
                                                       doc_id=doc_id,
                                                       data_type=data_type)

    # endregion