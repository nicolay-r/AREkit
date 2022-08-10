import collections
from collections import OrderedDict

from arekit.common.evaluation.result import BaseEvalResult
from arekit.common.labels.base import Label
from arekit.contrib.utils.evaluation.results.metrics_acc import calc_acc
from arekit.contrib.utils.evaluation.results.metrics_f1 import calc_f1_3c_macro, calc_f1_single_class
from arekit.contrib.utils.evaluation.results.metrics_pr import calc_precision_micro, calc_recall_micro, \
    calc_prec_and_recall


class ThreeClassPrecRecallF1EvalResult(BaseEvalResult):
    """ This evaluation considered both sentiment and non-sentiment (neutral).
    """

    C_F1 = 'f1'
    C_POS_PREC = 'pos_prec'
    C_NEG_PREC = 'neg_prec'
    C_NEU_PREC = 'neu_prec'
    C_POS_RECALL = 'pos_recall'
    C_NEG_RECALL = 'neg_recall'
    C_NEU_RECALL = 'neu_recall'
    C_PREC_MICRO = 'prec_micro'
    C_RECALL_MICRO = 'recall_micro'
    C_F1_POS = 'f1_pos'
    C_F1_NEG = 'f1_neg'
    C_F1_NEU = 'f1_neu'
    C_F1_MICRO = 'f1_micro'
    C_ACC = "acc"

    def __init__(self, label1, label2, label3, get_item_label_func):
        assert(isinstance(label1, Label))
        assert(isinstance(label2, Label))
        assert(isinstance(label3, Label))
        assert(callable(get_item_label_func))

        self.__pos_label = label1
        self.__neg_label = label2
        self.__neu_label = label3
        self.__get_item_label_func = get_item_label_func

        super(ThreeClassPrecRecallF1EvalResult, self).__init__(
            supported_labels={self.__pos_label, self.__neg_label, self.__neu_label})

        self.__doc_results = OrderedDict()

    def __has_opinions_with_label(self, data_items, label):
        assert(isinstance(data_items, collections.Iterable))
        assert(isinstance(label, Label))

        for item in data_items:
            if self.__get_item_label_func(item) == label:
                return True
        return False

    def reg_doc(self, cmp_pair, cmp_table):

        super(ThreeClassPrecRecallF1EvalResult, self).reg_doc(cmp_pair=cmp_pair,
                                                              cmp_table=cmp_table)

        has_pos = self.__has_opinions_with_label(
            data_items=cmp_pair.EtalonData,
            label=self.__pos_label)

        has_neg = self.__has_opinions_with_label(
            data_items=cmp_pair.EtalonData,
            label=self.__neg_label)

        has_neu = self.__has_opinions_with_label(
            data_items=cmp_pair.EtalonData,
            label=self.__neu_label)

        pos_prec, pos_recall = calc_prec_and_recall(cmp_table=cmp_table,
                                                    label=self.__pos_label,
                                                    opinions_exist=has_pos)

        neg_prec, neg_recall = calc_prec_and_recall(cmp_table=cmp_table,
                                                    label=self.__neg_label,
                                                    opinions_exist=has_neg)

        neu_prec, neu_recall = calc_prec_and_recall(cmp_table=cmp_table,
                                                    label=self.__neu_label,
                                                    opinions_exist=has_neu)

        # Add document results.
        f1 = calc_f1_3c_macro(pos_prec=pos_prec, neg_prec=neg_prec, neu_prec=neu_prec,
                              pos_recall=pos_recall, neg_recall=neg_recall, neu_recall=neu_recall)

        # Filling results.
        doc_id = cmp_pair.DocumentID
        self.__doc_results[doc_id] = OrderedDict()
        self.__doc_results[doc_id][self.C_F1] = f1
        self.__doc_results[doc_id][self.C_POS_PREC] = pos_prec
        self.__doc_results[doc_id][self.C_NEG_PREC] = neg_prec
        self.__doc_results[doc_id][self.C_NEU_PREC] = neu_prec
        self.__doc_results[doc_id][self.C_POS_RECALL] = pos_recall
        self.__doc_results[doc_id][self.C_NEG_RECALL] = neg_recall
        self.__doc_results[doc_id][self.C_NEU_RECALL] = neu_recall
        self.__doc_results[doc_id][self.C_PREC_MICRO] = calc_precision_micro(
            get_result_by_label_func=cmp_table.filter_result_column_by_label,
            labels=[self.__pos_label, self.__neg_label, self.__neu_label])
        self.__doc_results[doc_id][self.C_RECALL_MICRO] = calc_recall_micro(
            get_origin_answers_by_label_func=cmp_table.filter_original_column_by_label,
            get_result_answers_by_label_func=cmp_table.filter_result_column_by_label,
            labels=[self.__pos_label, self.__neg_label, self.__neu_label])
        self.__doc_results[doc_id][self.C_ACC] = calc_acc(cmp_table)

    def calculate(self):
        pos_prec_macro = 0.0
        neg_prec_macro = 0.0
        neu_prec_macro = 0.0
        pos_recall_macro = 0.0
        neg_recall_macro = 0.0
        neu_recall_macro = 0.0
        prec_micro_macro = 0.0
        recall_micro_macro = 0.0
        acc = 0.0

        for info in self.__doc_results.values():
            pos_prec_macro += info[self.C_POS_PREC]
            neg_prec_macro += info[self.C_NEG_PREC]
            neu_prec_macro += info[self.C_NEU_PREC]
            pos_recall_macro += info[self.C_POS_RECALL]
            neg_recall_macro += info[self.C_NEG_RECALL]
            neu_recall_macro += info[self.C_NEU_RECALL]
            prec_micro_macro += info[self.C_PREC_MICRO]
            recall_micro_macro += info[self.C_RECALL_MICRO]
            acc += info[self.C_ACC]

        if len(self.__doc_results) > 0:
            pos_prec_macro /= len(self.__doc_results)
            neg_prec_macro /= len(self.__doc_results)
            neu_prec_macro /= len(self.__doc_results)
            pos_recall_macro /= len(self.__doc_results)
            neg_recall_macro /= len(self.__doc_results)
            neu_recall_macro /= len(self.__doc_results)
            prec_micro_macro /= len(self.__doc_results)
            recall_micro_macro /= len(self.__doc_results)
            acc /= len(self.__doc_results)

        f1 = calc_f1_3c_macro(pos_prec=pos_prec_macro, neg_prec=neg_prec_macro, neu_prec=neu_prec_macro,
                              pos_recall=pos_recall_macro, neg_recall=neg_recall_macro, neu_recall=neu_recall_macro)

        # Filling total result.
        self._total_result[self.C_F1] = f1
        self._total_result[self.C_F1_POS] = calc_f1_single_class(prec=pos_prec_macro, recall=pos_recall_macro)
        self._total_result[self.C_F1_NEG] = calc_f1_single_class(prec=neg_prec_macro, recall=neg_recall_macro)
        self._total_result[self.C_F1_NEU] = calc_f1_single_class(prec=neu_prec_macro, recall=neu_recall_macro)
        self._total_result[self.C_POS_PREC] = pos_prec_macro
        self._total_result[self.C_NEG_PREC] = neg_prec_macro
        self._total_result[self.C_NEU_PREC] = neu_prec_macro
        self._total_result[self.C_POS_RECALL] = pos_recall_macro
        self._total_result[self.C_NEG_RECALL] = neg_recall_macro
        self._total_result[self.C_NEU_RECALL] = neu_recall_macro
        self._total_result[self.C_PREC_MICRO] = prec_micro_macro
        self._total_result[self.C_RECALL_MICRO] = recall_micro_macro
        self._total_result[self.C_F1_MICRO] = calc_f1_single_class(prec=prec_micro_macro, recall=recall_micro_macro)
        self._total_result[self.C_ACC] = acc

    def iter_document_results(self):
        return iter(self.__doc_results.items())
