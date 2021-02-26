from collections import OrderedDict

from arekit.common.evaluation.results import metrics
from arekit.common.evaluation.results.base import BaseEvalResult
from arekit.common.evaluation.results.utils import calc_f1_3c_macro, calc_f1_single_class
from arekit.common.labels.base import NegativeLabel, PositiveLabel, NeutralLabel, Label
from arekit.common.opinions.collection import OpinionCollection


class ThreeClassEvalResult(BaseEvalResult):
    """ This evaluation considered both sentiment and non-sentiment (neutral).
    """

    C_F1 = u'f1'
    C_POS_PREC = u'pos_prec'
    C_NEG_PREC = u'neg_prec'
    C_NEU_PREC = u'neu_prec'
    C_POS_RECALL = u'pos_recall'
    C_NEG_RECALL = u'neg_recall'
    C_NEU_RECALL = u'neu_recall'
    C_F1_POS = u'f1_pos'
    C_F1_NEG = u'f1_neg'
    C_F1_NEU = u'f1_neu'

    def __init__(self):
        super(ThreeClassEvalResult, self).__init__()
        self.__doc_results = OrderedDict()
        self.__pos_label = PositiveLabel()
        self.__neg_label = NegativeLabel()
        self.__neu_label = self.create_neutral_label()

    @staticmethod
    def create_neutral_label():
        return NeutralLabel()

    @staticmethod
    def __has_opinions_with_label(opinions, label):
        assert(isinstance(label, Label))
        assert(isinstance(opinions, OpinionCollection))
        for opinion in opinions:
            if opinion.Sentiment == label:
                return True
        return False

    def reg_doc(self, cmp_pair, cmp_table):

        super(ThreeClassEvalResult, self).reg_doc(cmp_pair=cmp_pair,
                                                  cmp_table=cmp_table)

        has_pos = self.__has_opinions_with_label(
            opinions=cmp_pair.EtalonOpinionCollection,
            label=self.__pos_label)

        has_neg = self.__has_opinions_with_label(
            opinions=cmp_pair.EtalonOpinionCollection,
            label=self.__neg_label)

        has_neu = self.__has_opinions_with_label(
            opinions=cmp_pair.EtalonOpinionCollection,
            label=self.__neu_label)

        pos_prec, pos_recall = metrics.calc_prec_and_recall(cmp_table=cmp_table,
                                                            label=self.__pos_label,
                                                            opinions_exist=has_pos)

        neg_prec, neg_recall = metrics.calc_prec_and_recall(cmp_table=cmp_table,
                                                            label=self.__neg_label,
                                                            opinions_exist=has_neg)

        neu_prec, neu_recall = metrics.calc_prec_and_recall(cmp_table=cmp_table,
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

    def calculate(self):
        pos_prec_macro, neg_prec_macro, neu_prec_macro, pos_recall_macro, neg_recall_macro, neu_recall_macro = \
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for info in self.__doc_results.itervalues():
            pos_prec_macro += info[self.C_POS_PREC]
            neg_prec_macro += info[self.C_NEG_PREC]
            neu_prec_macro += info[self.C_NEU_PREC]
            pos_recall_macro += info[self.C_POS_RECALL]
            neg_recall_macro += info[self.C_NEG_RECALL]
            neu_recall_macro += info[self.C_NEU_RECALL]

        if len(self.__doc_results) > 0:
            pos_prec_macro /= len(self.__doc_results)
            neg_prec_macro /= len(self.__doc_results)
            neu_prec_macro /= len(self.__doc_results)
            pos_recall_macro /= len(self.__doc_results)
            neg_recall_macro /= len(self.__doc_results)
            neu_recall_macro /= len(self.__doc_results)

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

    def iter_document_results(self):
        return self.__doc_results.iteritems()
