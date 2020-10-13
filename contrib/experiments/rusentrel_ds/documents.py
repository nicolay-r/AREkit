from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.contrib.experiments.ruattitudes.documents import RuAttitudesDocumentOperations
from arekit.contrib.experiments.rusentrel.documents import RuSentrelDocumentOperations


class RuSentrelWithRuAttitudesDocumentOperations(DocumentOperations):

    def __init__(self, rusentrel_doc, ruattitudes_doc, rusentrel_news_ids):
        assert(isinstance(rusentrel_doc, RuSentrelDocumentOperations))
        assert(isinstance(ruattitudes_doc, RuAttitudesDocumentOperations))
        self.__rusentrel_doc = rusentrel_doc
        self.__ruattitudes_doc = ruattitudes_doc
        self.__rusentrel_news_ids = rusentrel_news_ids

    # region CVBasedDocumentOperations

    def read_news(self, doc_id):
        if doc_id in self.__rusentrel_news_ids:
            return self.__rusentrel_doc.read_news(doc_id)
        return self.__ruattitudes_doc.read_news(doc_id)

    def iter_news_indices(self, data_type):
        for doc_id in self.__rusentrel_doc.iter_news_indices(data_type):
            yield doc_id

        for doc_id in self.__ruattitudes_doc.iter_news_indices(data_type):
            yield doc_id

    def iter_supported_data_types(self):
        yield DataType.Train
        yield DataType.Test

    # TODO. Weird
    def create_parse_options(self):
        return self.__rusentrel_doc.create_parse_options()

    # endregion