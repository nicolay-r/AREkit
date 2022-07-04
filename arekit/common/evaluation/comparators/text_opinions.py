from arekit.common.evaluation.comparators.base import BaseComparator
from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.evaluators.utils import check_is_supported, label_to_str
from arekit.common.text_opinions.base import TextOpinion


class TextOpinionBasedComparator(BaseComparator):
    """ Declared for `TextOpinionsToCompare`
    """

    def __init__(self, eval_mode):
        assert(isinstance(eval_mode, EvaluationModes))
        self.__eval_mode = eval_mode

    @staticmethod
    def text_opinion_to_id(text_opinion):
        """ Compose a unique opinion ID, based on the document information,
            and indices of the opinion participants.
        """
        assert(isinstance(text_opinion, TextOpinion))
        return "{}_{}_{}".format(text_opinion.DocID, text_opinion.SourceId, text_opinion.TargetId)

    @staticmethod
    def __create_index_by_id(etalon_text_opinions, id_func):
        index = {}
        for o_etalon in etalon_text_opinions:
            assert(isinstance(o_etalon, TextOpinion))
            index[id_func(o_etalon)] = o_etalon
        return index

    def __iter_diff_core(self, etalon_text_opins, test_text_opins):
        """ Perform the comparison by the exact
        """
        assert(isinstance(etalon_text_opins, list))
        assert(isinstance(test_text_opins, list))

        test_by_id = TextOpinionBasedComparator.__create_index_by_id(
            test_text_opins, id_func=self.text_opinion_to_id)

        etalon_by_id = TextOpinionBasedComparator.__create_index_by_id(
            etalon_text_opins, id_func=self.text_opinion_to_id)

        for o_etalon in etalon_text_opins:
            assert(isinstance(o_etalon, TextOpinion))
            o_id = self.text_opinion_to_id(o_etalon)
            o_test = test_by_id[o_id] if o_id in test_by_id else None
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

        for o_test in test_text_opins:
            assert (isinstance(o_test, TextOpinion))
            o_id = self.text_opinion_to_id(o_test)
            has_opinion = etalon_by_id[o_id] if o_id in etalon_by_id else None

            if has_opinion:
                # This case was covered by the prior loop.
                continue

            if self.__eval_mode == EvaluationModes.Classification:
                # That could not be possible, since we perform
                # classification of already provided opinions.
                raise Exception("Opinion of test collection (`{s}`->`{t}`) was not "
                                "found in etalon collection!".format(s=o_test.SourceId,
                                                                     t=o_test.TargetId))
            elif self.__eval_mode == EvaluationModes.Extraction:
                yield [o_test, None, o_test.Sentiment]

    def calc_diff(self, etalon, test, is_label_supported):
        """ Calculate the difference between a couple OpinionCollections
        """
        assert(isinstance(etalon, list))
        assert(isinstance(test, list))
        assert(callable(is_label_supported))

        it = self.__iter_diff_core(etalon_text_opins=etalon, test_text_opins=test)

        # Cache all rows into `rows` array
        rows = []
        for args in it:
            text_opinion, etalon_label, result_label = args

            check_is_supported(label=etalon_label, is_label_supported=is_label_supported)
            check_is_supported(label=result_label, is_label_supported=is_label_supported)

            row = ["source", "target",
                   None if etalon_label is None else label_to_str(etalon_label),
                   None if result_label is None else label_to_str(result_label),
                   self._cmp_result(l1=etalon_label, l2=result_label)]

            rows.append(row)

        return rows
