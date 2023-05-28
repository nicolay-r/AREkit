from arekit.common.experiment.api.ops_doc import DocumentOperations


class DictionaryBasedDocumentOperations(DocumentOperations):

    def __init__(self, d):
        assert(isinstance(d, dict))
        super(DictionaryBasedDocumentOperations, self).__init__()
        self.__d = d

    def by_id(self, doc_id):
        assert(isinstance(doc_id, int))
        return self.__d[doc_id]
