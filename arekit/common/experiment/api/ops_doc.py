from arekit.common.folding.base import BaseDataFolding
from arekit.common.text.parser import BaseTextParser


class DocumentOperations(object):
    """
    Provides operations with documents
    """

    def __init__(self, folding, text_parser=None):
        assert(isinstance(folding, BaseDataFolding))
        assert(isinstance(text_parser, BaseTextParser) or text_parser is None)
        self.__folding = folding
        self.__text_parser = text_parser

    # region properties

    @property
    def DataFolding(self):
        """ Algorithm, utilized in order to provide variety of foldings, such as
                cross-validation, fixed, none, etc.
        """
        return self.__folding

    # endregion

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
        data_types_splits = self.__folding.fold_doc_ids_set()

        if data_type not in data_types_splits:
            return
            yield

        for doc_id in data_types_splits[data_type]:
            yield doc_id

    def iter_parsed_docs(self, doc_ids):
        for doc_id in doc_ids:
            yield self.__parse_doc(doc_id=doc_id)

    def parse_doc(self, doc_id):
        return self.__parse_doc(doc_id=doc_id)

    # endregion

    # region private methods

    def __parse_doc(self, doc_id):
        news = self.get_doc(doc_id=doc_id)
        return self.__text_parser.parse_news(news=news)

    # endregion