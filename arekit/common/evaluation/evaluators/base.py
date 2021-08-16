import collections

from arekit.common.evaluation.cmp_opinions import OpinionCollectionsToCompare
from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable
from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.evaluators.utils import label_to_str
from arekit.common.evaluation.results.base import BaseEvalResult
from arekit.common.labels.base import Label
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection


class BaseEvaluator(object):

    def __init__(self, eval_mode):
        assert(isinstance(eval_mode, EvaluationModes))
        self.__eval_mode = eval_mode

    # region private methods

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

            o_test = test_opins.try_get_synonyms_opinion(o_etalon)
            has_opinion = o_test is not None

            if self.__eval_mode == EvaluationModes.Classification:
                # In case of evaluation mode, we do not consider such
                # cases when etalon opinion was not found in result.
                if not has_opinion:
                    continue
                # Otherwise provide the information for further comparison.
                yield [o_etalon, o_etalon.Sentiment, o_test.Sentiment]
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
                raise Exception("Opinion of test collection (`{s}`->`{t}`) was not "
                                "found in etalon collection!".format(s=o_test.SourceValue,
                                                                      t=o_test.TargetValue))
            elif self.__eval_mode == EvaluationModes.Extraction:
                yield [o_test, None, o_test.Sentiment]

    # endregion

    # region abstract methods

    def _create_eval_result(self):
        """ Provides instance which additionally performs all the necessary
            metrics-based computations
        """
        raise NotImplementedError()

    # endregion

    # region protected methods

    def _check_is_supported(self, label, is_label_supported):
        if label is None:
            return True

        if not is_label_supported(label):
            raise Exception("Label \"{label}\" is not supported by {e}".format(
                label=label_to_str(label),
                e=type(self).__name__))

    def _calc_diff(self, etalon_opins, test_opins, is_label_supported):
        assert(callable(is_label_supported))

        it = self.__iter_diff_core(etalon_opins=etalon_opins,
                                   test_opins=test_opins)

        # Cache all rows into `rows` array
        rows = []
        for args in it:
            opin, etalon_label, result_label = args

            self._check_is_supported(label=etalon_label, is_label_supported=is_label_supported)
            self._check_is_supported(label=result_label, is_label_supported=is_label_supported)

            row = [opin.SourceValue.encode('utf-8'),
                   opin.TargetValue.encode('utf-8'),
                   None if etalon_label is None else label_to_str(etalon_label),
                   None if result_label is None else label_to_str(result_label),
                   self.__cmp_result(l1=etalon_label, l2=result_label)]

            rows.append(row)

        # Filling dataframe.
        cmp_table = DocumentCompareTable.create_template_df(rows_count=len(rows))
        for o_ind, row in enumerate(rows):
            cmp_table.loc[o_ind] = row

        return DocumentCompareTable(cmp_table=cmp_table)

    # endregion

    def evaluate(self, cmp_pairs):
        assert (isinstance(cmp_pairs, collections.Iterable))

        # Composing result instance
        result = self._create_eval_result()
        assert(isinstance(result, BaseEvalResult))

        # Providing compared pairs in a form of tables.
        for cmp_pair in cmp_pairs:
            assert(isinstance(cmp_pair, OpinionCollectionsToCompare))
            cmp_table = self._calc_diff(etalon_opins=cmp_pair.EtalonOpinionCollection,
                                        test_opins=cmp_pair.TestOpinionCollection,
                                        is_label_supported=result.is_label_supported)

            result.reg_doc(cmp_pair=cmp_pair, cmp_table=cmp_table)

        # Calculate an overall result.
        result.calculate()

        return result
