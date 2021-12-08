from arekit.common.folding.base import BaseDataFolding
from arekit.processing.text.parser import DefaultTextParser


class DocumentOperations(object):
    """
    Provides operations with documents
    """

    def __init__(self, folding):
        assert(isinstance(folding, BaseDataFolding))
        self.__folding = folding
        self.__text_parser = DefaultTextParser()  # TODO. Temporary

    @property
    def DataFolding(self):
        """ Algorithm, utilized in order to provide variety of foldings, such as
                cross-validation, fixed, none, etc.
        """
        return self.__folding

    def get_doc(self, doc_id):
        raise NotImplementedError()

    def _get_text_parser(self):
        """ Default text parser instance.
            TODO. #219. It is expected to pass this instance as a parameter for the
            TODO. related methods. (Now it is limited due to the Annotator implmentation).
        """
        return self.__text_parser

    def iter_tagget_doc_ids(self, tag):
        """ Document identifiers which are grouped by a particular tag.
        """
        raise NotImplementedError()

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

    # TODO. This should be removed, since parse-options considered as a part of the text-parser instance!!!
    def _create_parse_options(self):
        raise NotImplementedError()

    def __parse_doc(self, doc_id):
        news = self.get_doc(doc_id=doc_id)
        text_parser = self._get_text_parser()
        return text_parser.parse_news(news=news, parse_options=self._create_parse_options())
