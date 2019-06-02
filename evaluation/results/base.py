import pandas as pd


class BaseEvalResult(object):

    def __init__(self):
        self.__documents = {}

    def get_cmp_table(self, doc_id):
        assert(isinstance(doc_id, int))
        return self.__documents[doc_id]

    def calculate(self):
        pass

    def add_cmp_table(self, doc_id, cmp_table):
        assert(doc_id not in self.__documents)
        assert(isinstance(cmp_table, pd.DataFrame))
        self.__documents[doc_id] = cmp_table

    def iter_document_cmp_tables(self):
        for doc_id, cmp_table in self.__documents.iteritems():
            yield doc_id, cmp_table

