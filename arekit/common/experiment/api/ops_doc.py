from arekit.common.folding.base import BaseDataFolding
from arekit.processing.text.parser import TextParser


class DocumentOperations(object):
    """
    Provides operations with documents
    """

    def __init__(self, folding):
        assert(isinstance(folding, BaseDataFolding))
        self.__folding = folding

    @property
    def DataFolding(self):
        """ Algorithm, utilized in order to provide variety of foldings, such as
                cross-validation, fixed, none, etc.
        """
        return self.__folding

    def get_doc(self, doc_id):
        raise NotImplementedError()

    # TODO. 212. Unify, add tag.
    def iter_doc_ids_to_annotate(self):
        """ provides set of documents that utilized by annotator algorithm in order to
            provide the related labeling of annotated attitudes in it.
            By default, we consider an empty set, so there is no need to utilize annotator.
        """
        raise NotImplementedError()

    # TODO. 212. Unify, add tag.
    def iter_doc_ids_to_compare(self):
        """ provides set of documents that utilized in model evaluation process
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
        return self.__parse_doc(doc_id)

    # TODO. This should be removed, since parse-options considered as a part
    # TODO. Of the text-parser instance!!!
    def _create_parse_options(self):
        raise NotImplementedError()

    def __parse_doc(self, doc_id):
        news = self.get_doc(doc_id=doc_id)
        # TODO. Use text parser as an instance.
        # TODO. (Current limitation: We depend on a particular text parser, which is not correct in general).
        return TextParser.parse_news(news=news,
                                     parse_options=self._create_parse_options())
