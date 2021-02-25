from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.contrib.experiments.ruattitudes.documents import RuAttitudesDocumentOperations
from arekit.contrib.experiments.rusentrel.documents import RuSentrelDocumentOperations


class RuSentrelWithRuAttitudesDocumentOperations(DocumentOperations):

    def __init__(self, rusentrel_doc, get_ruattitudes_doc):
        assert(isinstance(rusentrel_doc, RuSentrelDocumentOperations))
        assert(callable(get_ruattitudes_doc))

        # We consider RuSentRel folding algorithm by default.
        # The latter utilized in experiment as `main`, while
        # RuAttitude data-folding considered as `auxiliary`.
        super(RuSentrelWithRuAttitudesDocumentOperations, self).__init__(folding=rusentrel_doc.DataFolding)

        self.__rusentrel_doc = rusentrel_doc
        self.__get_ruattitudes_doc = get_ruattitudes_doc

    # region private methods

    def __select_doc_ops(self, doc_id):
        if self.__rusentrel_doc.DataFolding.contains_doc_id(doc_id):
            return self.__rusentrel_doc

        ruattitudes_doc = self.__get_ruattitudes_doc()
        assert(isinstance(ruattitudes_doc, RuAttitudesDocumentOperations))

        if ruattitudes_doc.DataFolding.contains_doc_id(doc_id):
            return ruattitudes_doc

        raise Exception(
            "Doc-id {} has not been found both in RuSentRel and RuAttitudes related doc formatters.".format(doc_id))

    # endregion

    # region DocumentOperations

    def read_news(self, doc_id):
        target_doc_ops = self.__select_doc_ops(doc_id)
        return target_doc_ops.read_news(doc_id)

    def iter_news_indices(self, data_type):
        for doc_id in self.__rusentrel_doc.iter_news_indices(data_type):
            yield doc_id

        ruattitudes_doc = self.__get_ruattitudes_doc()
        assert(isinstance(ruattitudes_doc, RuAttitudesDocumentOperations))

        for doc_id in ruattitudes_doc.iter_news_indices(data_type):
            yield doc_id

    def _create_parse_options(self):

        # ParseOptions are independent from doc_operations.
        # Therefore we provide rusentrel_doc by default.
        return self.__rusentrel_doc._create_parse_options()

    def iter_doc_ids_to_neutrally_annotate(self):
        return self.__rusentrel_doc.iter_doc_ids_to_neutrally_annotate()

    def iter_doc_ids_to_compare(self):
        return self.__rusentrel_doc.iter_doc_ids_to_compare()

    # endregion
