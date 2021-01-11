from collections import OrderedDict
from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable
from arekit.common.evaluation.results.base import BaseEvalResult
from arekit.common.evaluation.results.utils import calc_f1_single_class


class SingleClassEvalResult(BaseEvalResult):

    C_PREC = u'prec'
    C_RECALL = u'recall'

    def __init__(self):
        super(SingleClassEvalResult, self).__init__()

        self.__doc_results = OrderedDict()
        # TODO. To Base.
        self.__total_result = None

    # TODO. To Base.
    @property
    def TotalResult(self):
        return self.__total_result

    # TODO. To Base.
    def get_result_by_metric(self, metric_name):
        assert(isinstance(metric_name, unicode))
        return self.__total_result[metric_name]

    # TODO. To Base.
    def iter_total_by_param_results(self):
        assert(self.__total_result is not None)
        return self.__total_result.iteritems()

    def add_document_results(self, doc_id, cmp_table, prec, recall):
        assert(doc_id not in self.__doc_results)
        assert(isinstance(cmp_table, DocumentCompareTable))

        self._add_cmp_table(doc_id=doc_id, cmp_table=cmp_table)
        f1 = calc_f1_single_class(prec=prec, recall=recall)

        self.__doc_results[doc_id] = OrderedDict()
        self.__doc_results[doc_id][self.C_F1] = f1
        self.__doc_results[doc_id][self.C_PREC] = prec
        self.__doc_results[doc_id][self.C_RECALL] = recall

    def calculate(self):
        prec, recall = (0.0, 0.0)

        for info in self.__doc_results.itervalues():
            prec += info[self.C_PREC]
            recall += info[self.C_RECALL]

        prec /= len(self.__doc_results)
        recall /= len(self.__doc_results)

        f1 = calc_f1_single_class(prec=prec, recall=recall)

        self.__total_result = OrderedDict()
        self.__total_result[self.C_F1] = f1
        self.__total_result[self.C_PREC] = prec
        self.__total_result[self.C_RECALL] = recall
