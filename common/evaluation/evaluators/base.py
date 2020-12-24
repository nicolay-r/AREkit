from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable
from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.evaluators.utils import label_to_str
from arekit.common.labels.base import Label
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection


class BaseEvaluator(object):

    def __init__(self, eval_mode):
        assert(isinstance(eval_mode, EvaluationModes))
        self.__eval_mode = eval_mode

    @staticmethod
    def __cmp_result(l1, l2):
        assert(isinstance(l1, Label) or l1 is None)
        assert(isinstance(l2, Label) or l2 is None)

        if l1 is None or l2 is None:
            return False

        return l1 == l2

    def __iter_diff_core(self, etalon_opins, test_opins):
        assert(isinstance(etalon_opins, OpinionCollection))
        assert(isinstance(test_opins, OpinionCollection))

        for o_etalon in etalon_opins:
            assert(isinstance(o_etalon, Opinion))

            has_opinion = test_opins.has_synonymous_opinion(o_etalon)
            o_test = None if not has_opinion else test_opins.get_synonymous_opinion(o_etalon)

            if not has_opinion and self.__eval_mode == EvaluationModes.Classification:
                # In case of evaluation mode, we do not consider such
                # cases when etalon opinion was not found in result.
                continue
            elif self.__eval_mode == EvaluationModes.Extraction:
                yield [o_etalon,
                       o_etalon.Sentiment,
                       None if not has_opinion else o_test.Sentiment]

        for o_test in test_opins:
            assert(isinstance(o_test, Opinion))
            has_opinion = etalon_opins.has_synonymous_opinion(o_test)

            if has_opinion:
                # This case was covered by the prior loop.
                continue

            if self.__eval_mode == EvaluationModes.Classification:
                # That could not be possible, since we perform
                # classification of already provided opinions.
                raise Exception(u"Opinion of test collection (`{s}`->`{t}`) was not "
                                u"found in etalon collection!".format(s=o_test.SourceValue,
                                                                      t=o_test.TargetValue))
            elif self.__eval_mode == EvaluationModes.Extraction:
                yield [o_test, None, o_test.Sentiment]

    def evaluate(self, cmp_pairs):
        raise NotImplementedError()

    def calc_difference(self, etalon_opins, test_opins):

        cmp_table = DocumentCompareTable.create_template_df()

        it = self.__iter_diff_core(etalon_opins=etalon_opins,
                                   test_opins=test_opins)

        for o_ind, args in enumerate(it):
            opin, etalon_label, result_label = args
            cmp_table.loc[o_ind] = [opin.SourceValue.encode('utf-8'),
                                    opin.TargetValue.encode('utf-8'),
                                    None if etalon_label is None else label_to_str(etalon_label),
                                    None if result_label is None else label_to_str(result_label),
                                    self.__cmp_result(l1=etalon_label, l2=result_label)]

        return DocumentCompareTable(cmp_table=cmp_table)
