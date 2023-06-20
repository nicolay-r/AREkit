from arekit.common.data.doc_provider import DocumentProvider


class DictionaryBasedDocumentProvider(DocumentProvider):

    def __init__(self, d):
        assert(isinstance(d, dict))
        super(DictionaryBasedDocumentProvider, self).__init__()
        self.__d = d

    def by_id(self, doc_id):
        assert(isinstance(doc_id, int))
        return self.__d[doc_id]
