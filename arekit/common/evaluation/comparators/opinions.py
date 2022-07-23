from arekit.common.evaluation.comparators.base import BaseComparator
from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.evaluators.utils import label_to_str, check_is_supported
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection


class OpinionBasedComparator(BaseComparator):
    """ Performs a comparison of a couple OpinionCollections.
    """

    def __init__(self, eval_mode):
        assert(isinstance(eval_mode, EvaluationModes))
        self.__eval_mode = eval_mode

    # region private methods

    def __iter_diff_core(self, etalon_opins, test_opins):
        assert (isinstance(etalon_opins, OpinionCollection))
        assert (isinstance(test_opins, OpinionCollection))

        for o_etalon in etalon_opins:
            assert (isinstance(o_etalon, Opinion))

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
            assert (isinstance(o_test, Opinion))
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

    def calc_diff(self, etalon, test, is_label_supported):
        """ Calculate the difference between a couple OpinionCollections
        """
        assert(isinstance(etalon, OpinionCollection))
        assert(isinstance(test, OpinionCollection))
        assert (callable(is_label_supported))

        it = self.__iter_diff_core(etalon_opins=etalon, test_opins=test)

        # Cache all rows into `rows` array
        rows = []
        for args in it:
            opinion, etalon_label, result_label = args

            check_is_supported(label=etalon_label, is_label_supported=is_label_supported)
            check_is_supported(label=result_label, is_label_supported=is_label_supported)

            row = [None,
                   opinion.SourceValue,
                   opinion.TargetValue,
                   None if etalon_label is None else label_to_str(etalon_label),
                   None if result_label is None else label_to_str(result_label),
                   self._cmp_result(l1=etalon_label, l2=result_label)]

            rows.append(row)

        return rows
