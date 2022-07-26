from arekit.common.evaluation.comparators.base import BaseComparator
from arekit.common.evaluation.context_opinion import ContextOpinion
from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.evaluators.utils import check_is_supported, label_to_str


class TextOpinionBasedComparator(BaseComparator):
    """ Declared for `TextOpinionsToCompare`
    """

    def __init__(self, eval_mode):
        assert(isinstance(eval_mode, EvaluationModes))
        self.__eval_mode = eval_mode

    @staticmethod
    def context_opinion_to_id(context_opinion):
        """ Compose a unique opinion ID, based on the document information,
            and indices of the opinion participants.
        """
        assert(isinstance(context_opinion, ContextOpinion))
        return "{doc_id}_{ctx_id}_{src_id}_{tgt_id}".format(
            doc_id=context_opinion.DocId,
            ctx_id=context_opinion.ContextId,
            src_id=context_opinion.SourceId,
            tgt_id=context_opinion.TargetId)

    @staticmethod
    def __create_index_by_id(context_opinions, id_func):
        index = {}
        for o_etalon in context_opinions:
            assert(isinstance(o_etalon, ContextOpinion))
            index[id_func(o_etalon)] = o_etalon
        return index

    def __iter_diff_core(self, etalon_context_opinions, test_context_opinions, id_func):
        """ Perform the comparison by the exact
        """
        assert(isinstance(etalon_context_opinions, list))
        assert(isinstance(test_context_opinions, list))
        assert(callable(id_func))

        test_by_id = TextOpinionBasedComparator.__create_index_by_id(
            test_context_opinions, id_func=id_func)

        etalon_by_id = TextOpinionBasedComparator.__create_index_by_id(
            etalon_context_opinions, id_func=id_func)

        for o_etalon in etalon_context_opinions:
            assert(isinstance(o_etalon, ContextOpinion))
            o_id = id_func(o_etalon)
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

        for o_test in test_context_opinions:
            assert(isinstance(o_test, ContextOpinion))
            o_id = id_func(o_test)
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

        it = self.__iter_diff_core(etalon_context_opinions=etalon,
                                   test_context_opinions=test,
                                   id_func=self.context_opinion_to_id)

        # Cache all rows into `rows` array
        rows = []
        for args in it:
            text_opinion, etalon_label, result_label = args

            check_is_supported(label=etalon_label, is_label_supported=is_label_supported)
            check_is_supported(label=result_label, is_label_supported=is_label_supported)

            row = [self.context_opinion_to_id(text_opinion),
                   "source",
                   "target",
                   None if etalon_label is None else label_to_str(etalon_label),
                   None if result_label is None else label_to_str(result_label),
                   self._cmp_result(l1=etalon_label, l2=result_label)]

            rows.append(row)

        return rows
