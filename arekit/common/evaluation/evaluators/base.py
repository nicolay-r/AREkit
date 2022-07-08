import collections

from arekit.common.evaluation.comparators.base import BaseComparator
from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable
from arekit.common.evaluation.pairs.base import BasePairToCompare
from arekit.common.evaluation.result import BaseEvalResult


class BaseEvaluator(object):

    def __init__(self, comparator):
        assert(isinstance(comparator, BaseComparator))
        self.__comp = comparator

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
        rows = self.__comp.calc_diff(etalon=etalon_data, test=test_data, is_label_supported=is_label_supported)

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
            assert(isinstance(cmp_pair, BasePairToCompare))
            cmp_table = self._calc_diff(etalon_data=cmp_pair.EtalonData,
                                        test_data=cmp_pair.TestData,
                                        is_label_supported=result.is_label_supported)

            result.reg_doc(cmp_pair=cmp_pair, cmp_table=cmp_table)

        # Calculate an overall result.
        result.calculate()

        return result
