from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable


class BaseEvalResult(object):

    C_F1 = u'f1'

    def __init__(self):
        self._cmp_tables = {}

    def get_result_by_metric(self, metric_name):
        raise NotImplementedError()

    def get_result_as_str(self):
        raise NotImplementedError()

    def calculate(self):
        raise NotImplementedError()

    def _add_cmp_table(self, doc_id, cmp_table):
        assert(doc_id not in self._cmp_tables)
        assert(isinstance(cmp_table, DocumentCompareTable))
        self._cmp_tables[doc_id] = cmp_table
