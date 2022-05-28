from arekit.common.experiment.api.ctx_base import ExperimentContext
from arekit.common.news.parser import NewsParser
from arekit.common.text.parser import BaseTextParser


class DocumentOperations(object):
    """
    Provides operations with documents
    """

    def __init__(self, exp_ctx, text_parser=None):
        assert(isinstance(exp_ctx, ExperimentContext) or exp_ctx is None)
        assert(isinstance(text_parser, BaseTextParser) or text_parser is None)
        self._exp_ctx = exp_ctx
        self.__text_parser = text_parser

    # region abstract methods

    def get_doc(self, doc_id):
        raise NotImplementedError()

    def iter_tagget_doc_ids(self, tag):
        """ Document identifiers which are grouped by a particular tag.
        """
        raise NotImplementedError()

    # endregion

    # region public methods

    def iter_doc_ids(self, data_type):
        """ Provides a news indices, related to a particular `data_type`
        """
        data_types_splits = self._exp_ctx.DataFolding.fold_doc_ids_set()

        if data_type not in data_types_splits:
            return
            yield

        for doc_id in data_types_splits[data_type]:
            yield doc_id

    def parse_doc(self, doc_id):
        return self.__parse_doc(doc_id=doc_id)

    # endregion

    # region private methods

    def __parse_doc(self, doc_id):
        news = self.get_doc(doc_id=doc_id)
        return NewsParser.parse(news=news, text_parser=self.__text_parser)

    # endregion