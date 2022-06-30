import collections

from arekit.common.evaluation.calc.opinions import OpinionsComparisonCalculator
from arekit.common.evaluation.cmp_opinions import OpinionCollectionsToCompare
from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable
from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.results.base import BaseEvalResult


class BaseEvaluator(object):

    def __init__(self, eval_mode):
        assert(isinstance(eval_mode, EvaluationModes))
        self.__calc = OpinionsComparisonCalculator(eval_mode)

    # region abstract methods

    def _create_eval_result(self):
        """ Provides instance which additionally performs all the necessary
            metrics-based computations
        """
        raise NotImplementedError()

    # endregion

    # region protected methods

    def _calc_diff(self, etalon_data, test_data, is_label_supported):
        assert(callable(is_label_supported))

        # Obtaining comparison rows.
        rows = self.__calc.calc_diff(etalon=etalon_data, test=test_data, is_label_supported=is_label_supported)

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
            cmp_table = self._calc_diff(etalon_data=cmp_pair.EtalonOpinionCollection,
                                        test_data=cmp_pair.TestOpinionCollection,
                                        is_label_supported=result.is_label_supported)

            result.reg_doc(cmp_pair=cmp_pair, cmp_table=cmp_table)

        # Calculate an overall result.
        result.calculate()

        return result
