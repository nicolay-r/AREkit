import pandas as pd
from core.evaluation.labels import Label
from core.evaluation.utils import FilesToCompare
from core.source.opinion import OpinionCollection
from core.source.synonyms import SynonymsCollection


class BaseEvaluator(object):

    C_WHO = 'who'
    C_TO = 'to'
    C_ORIG = 'how_orig'
    C_RES = 'how_results'
    C_CMP = 'comparison'

    def __init__(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection) and synonyms.IsReadOnly)
        self.__synonyms = synonyms

    @property
    def Synonyms(self):
        return self.__synonyms

    @staticmethod
    def calc_recall(results, answers, label, answer_exist):
        assert(isinstance(results, pd.DataFrame))
        assert(isinstance(label, Label))
        assert(isinstance(answer_exist, bool))
        total = len(results[results[BaseEvaluator.C_ORIG] == label.to_str()])
        if total != 0:
            return 1.0 * len(answers[(answers[BaseEvaluator.C_CMP] == True)]) / total
        else:
            return 0.0 if answer_exist else 1.0

    @staticmethod
    def calc_precision(answers, answer_exist):
        assert(isinstance(answers, pd.DataFrame))
        assert(isinstance(answer_exist, bool))
        total = len(answers)
        if total != 0:
            return 1.0 * len(answers[(answers[BaseEvaluator.C_CMP] == True)]) / total
        else:
            return 0.0 if answer_exist else 1.0

    def calc_difference(self, etalon_opins, test_opins):
        assert(isinstance(etalon_opins, OpinionCollection))
        assert(isinstance(test_opins, OpinionCollection))

        df = pd.DataFrame(
            columns=[self.C_WHO, self.C_TO, self.C_ORIG, self.C_RES, self.C_CMP])

        r_ind = 0
        for o_etalon in etalon_opins:
            comparison = False
            has_opinion = test_opins.has_synonymous_opinion(o_etalon)
            if has_opinion:
                o_test = test_opins.get_synonymous_opinion(o_etalon)
                comparison = o_test.sentiment == o_etalon.sentiment

            df.loc[r_ind] = [o_etalon.value_left.encode('utf-8'),
                             o_etalon.value_right.encode('utf-8'),
                             o_etalon.sentiment.to_str(),
                             None if not has_opinion else o_test.sentiment.to_str(),
                             comparison]
            r_ind += 1

        for o_test in test_opins:
            has_opinion = etalon_opins.has_synonymous_opinion(o_test)
            if has_opinion:
                continue
            df.loc[r_ind] = [o_test.value_left.encode('utf-8'),
                             o_test.value_right.encode('utf-8'),
                             None, o_test.sentiment.to_str(), False]
            r_ind += 1

        return df

    def calc_a_file(self, files_to_compare, debug):
        assert(isinstance(files_to_compare, FilesToCompare))

        # Reading test answers.
        test_opins = OpinionCollection.from_file(
            filepath=files_to_compare.TestFilepath,
            synonyms=self.__synonyms)

        # Reading etalon answers.
        etalon_opins = OpinionCollection.from_file(
            filepath=files_to_compare.EtalonFilepath,
            synonyms=self.__synonyms)

        if debug:
            print "{} <-> {}, {}".format(
                files_to_compare.TestFilepath,
                files_to_compare.EtalonFilepath,
                files_to_compare.index)

        return test_opins, etalon_opins

    @staticmethod
    def calc_prec_and_recall(results, label, opinions_exist):
        assert(isinstance(opinions_exist, bool))
        assert(isinstance(label, Label))

        answers = results[(results[BaseEvaluator.C_RES] == label.to_str())]
        p = BaseEvaluator.calc_precision(answers, answer_exist=opinions_exist)
        r = BaseEvaluator.calc_recall(results, answers, label, answer_exist=opinions_exist)

        assert(isinstance(p, float))
        assert(isinstance(r, float))

        return p, r
