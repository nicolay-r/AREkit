from collections import OrderedDict

from arekit.common.evaluation.cmp_opinions import OpinionCollectionsToCompare
from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable


class BaseEvalResult(object):

    def __init__(self):
        self._cmp_tables = {}
        self._total_result = OrderedDict()

    # region properties

    @property
    def TotalResult(self):
        return self._total_result

    # endregion

    # region abstract methods

    def calculate(self):
        raise NotImplementedError()

    # endregion

    def get_result_by_metric(self, metric_name):
        assert(isinstance(metric_name, unicode))
        return self._total_result[metric_name]

    def iter_total_by_param_results(self):
        assert(self._total_result is not None)
        return self._total_result.iteritems()

    def iter_dataframe_cmp_tables(self):
        yield self._cmp_tables.iteritems()

    def reg_doc(self, cmp_pair, cmp_table):
        """ Registering cmp_table.
        """
        assert(isinstance(cmp_pair, OpinionCollectionsToCompare))
        assert(isinstance(cmp_table, DocumentCompareTable))
        self._cmp_tables[cmp_pair.DocumentID] = cmp_table
