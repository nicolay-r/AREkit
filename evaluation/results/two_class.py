from collections import OrderedDict

from arekit.evaluation.evaluators.cmp_table import DocumentCompareTable
from arekit.evaluation.results.base import BaseEvalResult
from arekit.evaluation.results.utils import calc_f1_single_class, calc_f1


class TwoClassEvalResult(BaseEvalResult):

    C_POS_PREC = u'pos_prec'
    C_NEG_PREC = u'neg_prec'
    C_POS_RECALL = u'pos_recall'
    C_NEG_RECALL = u'neg_recall'
    C_F1_POS = u'f1_pos'
    C_F1_NEG = u'f1_neg'
    C_F1 = u'f1'

    def __init__(self):
        super(TwoClassEvalResult, self).__init__()

        self.__documents = OrderedDict()
        self.__cmp_results = OrderedDict()
        self.__result = None

    def get_result_as_str(self):
        return str(self.__result)

    def get_result_by_metric(self, metric_name):
        assert(isinstance(metric_name, unicode))
        return self.__result[metric_name]

    def iter_results(self):
        assert(self.__result is not None)
        for metric_name, value in self.__result.iteritems():
            yield metric_name, value

    def add_document_results(self, doc_id,
                             cmp_table,
                             pos_prec, neg_prec,
                             pos_recall, neg_recall):
        assert(doc_id not in self.__documents)
        assert(isinstance(cmp_table, DocumentCompareTable))

        self.add_cmp_table(doc_id=doc_id, cmp_table=cmp_table)

        f1 = calc_f1(pos_prec=pos_prec,
                     neg_prec=neg_prec,
                     pos_recall=pos_recall,
                     neg_recall=neg_recall)

        self.__documents[doc_id] = OrderedDict()
        self.__documents[doc_id][self.C_F1] = round(f1, 2)
        self.__documents[doc_id][self.C_POS_PREC] = round(pos_prec, 4)
        self.__documents[doc_id][self.C_NEG_PREC] = round(neg_prec, 5)
        self.__documents[doc_id][self.C_POS_RECALL] = round(pos_recall, 5)
        self.__documents[doc_id][self.C_NEG_RECALL] = round(neg_recall, 5)

    def add_cmp_results(self, doc_id, cmp_results):
        assert(doc_id not in self.__cmp_results)
        self.__cmp_results[doc_id] = cmp_results

    def calculate(self):
        pos_prec, neg_prec, pos_recall, neg_recall = (0.0, 0.0, 0.0, 0.0)

        for info in self.__documents.itervalues():
            pos_prec += info[self.C_POS_PREC]
            neg_prec += info[self.C_NEG_PREC]
            pos_recall += info[self.C_POS_RECALL]
            neg_recall += info[self.C_NEG_RECALL]

        if len(self.__documents) > 0:
            pos_prec /= len(self.__documents)
            neg_prec /= len(self.__documents)
            pos_recall /= len(self.__documents)
            neg_recall /= len(self.__documents)

        f1 = calc_f1(pos_prec=pos_prec,
                     neg_prec=neg_prec,
                     pos_recall=pos_recall,
                     neg_recall=neg_recall)

        self.__result = OrderedDict()
        self.__result[self.C_F1] = f1
        self.__result[self.C_F1_POS] = calc_f1_single_class(prec=pos_prec, recall=pos_recall)
        self.__result[self.C_F1_NEG] = calc_f1_single_class(prec=neg_prec, recall=neg_recall)
        self.__result[self.C_POS_PREC] = pos_prec
        self.__result[self.C_NEG_PREC] = neg_prec
        self.__result[self.C_POS_RECALL] = pos_recall
        self.__result[self.C_NEG_RECALL] = neg_recall

    def iter_document_results(self):
        for doc_id, info in self.__documents.iteritems():
            yield doc_id, info

    def iter_document_cmp(self):
        for doc_id, cmp_result in self.__cmp_results.iteritems():
            yield doc_id, cmp_result
