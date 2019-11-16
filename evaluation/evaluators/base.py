from core.common.opinions.base import Opinion
from core.common.opinions.collection import OpinionCollection
from core.common.synonyms import SynonymsCollection
from core.evaluation.evaluators.cmp_table import DocumentCompareTable


class BaseEvaluator(object):

    def __init__(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection) and synonyms.IsReadOnly)
        self.__synonyms = synonyms

    def evaluate(self, cmp_pairs):
        raise Exception("Not implemented")

    @property
    def Synonyms(self):
        return self.__synonyms

    def calc_difference(self, etalon_opins, test_opins):
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
                                    o_etalon.Sentiment.to_str(),
                                    None if not has_opinion else o_test.Sentiment.to_str(),
                                    comparison]
            r_ind += 1

        for o_test in test_opins:
            assert(isinstance(o_test, Opinion))
            has_opinion = etalon_opins.has_synonymous_opinion(o_test)
            if has_opinion:
                continue
            cmp_table.loc[r_ind] = [o_test.SourceValue.encode('utf-8'),
                                    o_test.TargetValue.encode('utf-8'),
                                    None, o_test.Sentiment.to_str(), False]
            r_ind += 1

        return DocumentCompareTable(cmp_table=cmp_table)
