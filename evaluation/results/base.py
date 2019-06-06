import pandas as pd


# TODO. Move into separated file (eval/cmp_table.py).
class DocumentCompareTable:

    # TODO. Move column names here (from eval).
    # TODO. Move filtering operations here.

    def __init__(self, cmp_table):
        assert(isinstance(cmp_table, pd.DataFrame))
        self.__cmp_table = cmp_table

    def load(self, filepath):
        assert(isinstance(filepath, unicode))
        self.__cmp_table.from_csv(filepath)

    def save(self, filepath):
        assert(isinstance(filepath, unicode))
        self.__cmp_table.to_csv(filepath)


class BaseEvalResult(object):

    def __init__(self):
        self.__documents = {}

    def get_cmp_table(self, doc_id):
        assert(isinstance(doc_id, int))
        return self.__documents[doc_id]

    def calculate(self):
        raise Exception("Not implemented")

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
