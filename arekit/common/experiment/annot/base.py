import logging

from arekit.common.experiment.api.enums import BaseDocumentTag
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

    # region private methods

    def __iter_annotated_collections(self, data_type, doc_ops, opin_ops):
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(opin_ops, OpinionOperations))

        logged_parsed_news_iter = progress_bar_iter(
            iterable=doc_ops.iter_parsed_docs(doc_ops.iter_tagget_doc_ids(BaseDocumentTag.Annotate)),
            desc="Annotating parsed news [{}]".format(data_type))

        for parsed_news in logged_parsed_news_iter:
            assert(isinstance(parsed_news, ParsedNews))
            yield parsed_news.RelatedDocID, \
                  self._annot_collection_core(parsed_news=parsed_news, data_type=data_type, opin_ops=opin_ops)

    # endregion

    def _annot_collection_core(self, parsed_news, data_type, opin_ops):
        raise NotImplementedError

    # region public methods

    def iter_annotated_collections(self, data_type, doc_ops, opin_ops):
        assert(isinstance(opin_ops, OpinionOperations))
        return self.__iter_annotated_collections(data_type=data_type,
                                                 doc_ops=doc_ops,
                                                 opin_ops=opin_ops)

    # endregion