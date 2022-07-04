from collections import OrderedDict

from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable
from arekit.common.evaluation.pairs.base import BasePairToCompare


class BaseEvalResult(object):

    def __init__(self, supported_labels):
        assert(isinstance(supported_labels, set))
        self._cmp_tables = {}
        self._total_result = OrderedDict()
        self.__supported_labels = supported_labels

    # region properties

    @property
    def TotalResult(self):
        return self._total_result

    # endregion

    # region abstract methods

    def calculate(self):
        raise NotImplementedError()

    # endregion

    def is_label_supported(self, label):
        return label in self.__supported_labels

    def get_result_by_metric(self, metric_name):
        assert(isinstance(metric_name, str))
        return self._total_result[metric_name]

    def iter_total_by_param_results(self):
        assert(self._total_result is not None)
        return iter(self._total_result.items())

    def iter_dataframe_cmp_tables(self):
        return iter(self._cmp_tables.items())

    def reg_doc(self, cmp_pair, cmp_table):
        """ Registering cmp_table.
        """
        assert(isinstance(cmp_pair, BasePairToCompare))
        assert(isinstance(cmp_table, DocumentCompareTable))
        self._cmp_tables[cmp_pair.DocumentID] = cmp_table
