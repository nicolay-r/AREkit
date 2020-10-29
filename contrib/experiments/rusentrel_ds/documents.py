from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.contrib.experiments.ruattitudes.documents import RuAttitudesDocumentOperations
from arekit.contrib.experiments.rusentrel.documents import RuSentrelDocumentOperations


class RuSentrelWithRuAttitudesDocumentOperations(DocumentOperations):

    def __init__(self, rusentrel_doc, ruattitudes_doc):
        assert(isinstance(rusentrel_doc, RuSentrelDocumentOperations))
        assert(isinstance(ruattitudes_doc, RuAttitudesDocumentOperations))
        self.__rusentrel_doc = rusentrel_doc
        self.__ruattitudes_doc = ruattitudes_doc

    # region CVBasedDocumentOperations

    def read_news(self, doc_id):
        doc = self.__rusentrel_doc if self.__rusentrel_doc.contains_doc_id(doc_id) else self.__ruattitudes_doc
        return doc.read_news(doc_id)

    def iter_news_indices(self, data_type):
        for doc_id in self.__rusentrel_doc.iter_news_indices(data_type):
            yield doc_id

        for doc_id in self.__ruattitudes_doc.iter_news_indices(data_type):
            yield doc_id

    def iter_supported_data_types(self):
        yield DataType.Train
        yield DataType.Test

    def _create_parse_options(self):
        return self.__rusentrel_doc._create_parse_options()

    def get_doc_ids_set_to_neutrally_annotate(self):
        return self.__rusentrel_doc.get_doc_ids_set_to_neutrally_annotate()

    # endregion