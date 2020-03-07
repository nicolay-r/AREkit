from arekit.common.evaluation import DocumentCompareTable


class BaseEvalResult(object):

    def __init__(self):
        self.__documents = {}

    def get_cmp_table(self, doc_id):
        assert(isinstance(doc_id, int))
        return self.__documents[doc_id]

    def calculate(self):
        raise NotImplementedError()

    def add_cmp_table(self, doc_id, cmp_table):
        assert(doc_id not in self.__documents)
        assert(isinstance(cmp_table, DocumentCompareTable))
        self.__documents[doc_id] = cmp_table

    def iter_document_cmp_tables(self):
        for doc_id, cmp_table in self.__documents.iteritems():
            yield doc_id, cmp_table

    def iter_document_ids(self):
        for doc_id in self.__documents.iterkeys():
            yield doc_id
