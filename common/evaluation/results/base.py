from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable


class BaseEvalResult(object):

    def __init__(self):
        self.__cmp_tables = {}

    def get_result_as_str(self):
        raise NotImplementedError()

    def calculate(self):
        raise NotImplementedError()

    def _add_cmp_table(self, doc_id, cmp_table):
        assert(doc_id not in self.__cmp_tables)
        assert(isinstance(cmp_table, DocumentCompareTable))
        self.__cmp_tables[doc_id] = cmp_table
