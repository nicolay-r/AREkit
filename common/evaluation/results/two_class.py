from collections import OrderedDict

from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable
from arekit.common.evaluation.results.base import BaseEvalResult
from arekit.common.evaluation.results.utils import calc_f1_single_class, calc_f1


class TwoClassEvalResult(BaseEvalResult):

    C_POS_PREC = u'pos_prec'
    C_NEG_PREC = u'neg_prec'
    C_POS_RECALL = u'pos_recall'
    C_NEG_RECALL = u'neg_recall'
    C_F1_POS = u'f1_pos'
    C_F1_NEG = u'f1_neg'

    def __init__(self):
        super(TwoClassEvalResult, self).__init__()

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

    def add_document_results(self, doc_id,
                             cmp_table,
                             pos_prec, neg_prec,
                             pos_recall, neg_recall):
        assert(doc_id not in self.__doc_results)
        assert(isinstance(cmp_table, DocumentCompareTable))

        self._add_cmp_table(doc_id=doc_id, cmp_table=cmp_table)

        f1 = calc_f1(pos_prec=pos_prec,
                     neg_prec=neg_prec,
                     pos_recall=pos_recall,
                     neg_recall=neg_recall)

        self.__doc_results[doc_id] = OrderedDict()
        self.__doc_results[doc_id][self.C_F1] = f1
        self.__doc_results[doc_id][self.C_POS_PREC] = pos_prec
        self.__doc_results[doc_id][self.C_NEG_PREC] = neg_prec
        self.__doc_results[doc_id][self.C_POS_RECALL] = pos_recall
        self.__doc_results[doc_id][self.C_NEG_RECALL] = neg_recall

    def calculate(self):
        pos_prec, neg_prec, pos_recall, neg_recall = (0.0, 0.0, 0.0, 0.0)

        for info in self.__doc_results.itervalues():
            pos_prec += info[self.C_POS_PREC]
            neg_prec += info[self.C_NEG_PREC]
            pos_recall += info[self.C_POS_RECALL]
            neg_recall += info[self.C_NEG_RECALL]

        if len(self.__doc_results) > 0:
            pos_prec /= len(self.__doc_results)
            neg_prec /= len(self.__doc_results)
            pos_recall /= len(self.__doc_results)
            neg_recall /= len(self.__doc_results)

        f1 = calc_f1(pos_prec=pos_prec,
                     neg_prec=neg_prec,
                     pos_recall=pos_recall,
                     neg_recall=neg_recall)

        self.__total_result = OrderedDict()
        self.__total_result[self.C_F1] = f1
        self.__total_result[self.C_F1_POS] = calc_f1_single_class(prec=pos_prec, recall=pos_recall)
        self.__total_result[self.C_F1_NEG] = calc_f1_single_class(prec=neg_prec, recall=neg_recall)
        self.__total_result[self.C_POS_PREC] = pos_prec
        self.__total_result[self.C_NEG_PREC] = neg_prec
        self.__total_result[self.C_POS_RECALL] = pos_recall
        self.__total_result[self.C_NEG_RECALL] = neg_recall

    def iter_document_results(self):
        return self.__doc_results.iteritems()
