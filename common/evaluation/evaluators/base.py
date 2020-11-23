from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable
from arekit.common.evaluation.evaluators.utils import label_to_str
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection


class BaseEvaluator(object):

    def evaluate(self, cmp_pairs):
        raise NotImplementedError()

    @staticmethod
    def calc_difference(etalon_opins, test_opins):
        assert(isinstance(etalon_opins, OpinionCollection))
        assert(isinstance(test_opins, OpinionCollection))

        cmp_table = DocumentCompareTable.create_template_df()

        r_ind = 0
        for o_etalon in etalon_opins:
            assert(isinstance(o_etalon, Opinion))

            comparison = False
            has_opinion = test_opins.has_synonymous_opinion(o_etalon)
            if has_opinion:
                o_test = test_opins.get_synonymous_opinion(o_etalon)
                assert(isinstance(o_test, Opinion))
                comparison = o_test.Sentiment == o_etalon.Sentiment

            cmp_table.loc[r_ind] = [o_etalon.SourceValue.encode('utf-8'),
                                    o_etalon.TargetValue.encode('utf-8'),
                                    label_to_str(o_etalon.Sentiment),
                                    None if not has_opinion else label_to_str(o_test.Sentiment),
                                    comparison]
            r_ind += 1

        for o_test in test_opins:
            assert(isinstance(o_test, Opinion))
            has_opinion = etalon_opins.has_synonymous_opinion(o_test)
            if has_opinion:
                continue
            cmp_table.loc[r_ind] = [o_test.SourceValue.encode('utf-8'),
                                    o_test.TargetValue.encode('utf-8'),
                                    None, label_to_str(o_test.Sentiment), False]
            r_ind += 1

        return DocumentCompareTable(cmp_table=cmp_table)
