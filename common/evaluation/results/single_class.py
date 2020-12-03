from collections import OrderedDict
from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable
from arekit.common.evaluation.results.base import BaseEvalResult
from arekit.common.evaluation.results.utils import calc_f1_single_class


class SingleClassEvalResult(BaseEvalResult):

    C_PREC = u'prec'
    C_RECALL = u'recall'
    C_F1 = u'f1'

    def __init__(self):
        super(SingleClassEvalResult, self).__init__()

        self.__documents = OrderedDict()
        self.__result = None

    @property
    def Result(self):
        return self.__result

    def add_document_results(self, doc_id, cmp_table, prec, recall):
        assert(doc_id not in self.__documents)
        assert(isinstance(cmp_table, DocumentCompareTable))

        self._add_cmp_table(doc_id=doc_id, cmp_table=cmp_table)
        f1 = calc_f1_single_class(prec=prec, recall=recall)

        self.__documents[doc_id] = OrderedDict()
        self.__documents[doc_id][self.C_F1] = round(f1, 2)
        self.__documents[doc_id][self.C_PREC] = round(prec, 4)
        self.__documents[doc_id][self.C_RECALL] = round(recall, 5)

    def calculate(self):
        prec, recall = (0.0, 0.0)

        for info in self.__documents.itervalues():
            prec += info[self.C_PREC]
            recall += info[self.C_RECALL]

        prec /= len(self.__documents)
        recall /= len(self.__documents)

        f1 = calc_f1_single_class(prec=prec, recall=recall)

        self.__result = OrderedDict()
        self.__result[self.C_F1] = f1
        self.__result[self.C_PREC] = prec
        self.__result[self.C_RECALL] = recall
