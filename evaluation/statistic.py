
import pandas as pd

from core.evaluation.labels import Label, PositiveLabel, NegativeLabel
from core.processing.lemmatization.base import Stemmer
from core.source.opinion import OpinionCollection


class MethodStatistic:

    @staticmethod
    def founded_opins(test_opins, etalon_opins, sentiment=None):
        assert(isinstance(test_opins, OpinionCollection))
        assert(isinstance(etalon_opins, OpinionCollection))
        assert(isinstance(sentiment, Label) or sentiment is None)

        founded = 0
        for opin in test_opins:
            if sentiment is not None:
                if sentiment != opin.sentiment:
                    continue
            if etalon_opins.has_opinion_by_synonyms(opin, sentiment):
                founded += 1

        return founded

    @staticmethod
    def get_method_statistic(files_to_compare_list, synonyms_filepath, stemmer):
        """
            Calculate statistic based on result files
            files_to_compare_list: list
                list of FilesToCompare objects
            synonyms_filepath: str
            stemmer: Stemmer
        """
        assert(isinstance(stemmer, Stemmer))

        columns = ["t_all", "t_pos", "t_neg", "e_all", "e_pos", "e_neg"]

        df = pd.DataFrame(columns=columns)
        for files_to_compare in files_to_compare_list:

            assert(isinstance(files_to_compare, FilesToCompare))
            test_opins = OpinionCollection.from_file(
                    files_to_compare.test_filepath, synonyms_filepath, stemmer=stemmer)
            etalon_opins = OpinionCollection.from_file(
                    files_to_compare.etalon_filepath, synonyms_filepath, stemmer=stemmer)

            df.loc[files_to_compare.index] = [
                    MethodStatistic.founded_opins(test_opins, etalon_opins),
                    MethodStatistic.founded_opins(test_opins, etalon_opins, PositiveLabel()),
                    MethodStatistic.founded_opins(test_opins, etalon_opins, NegativeLabel()),
                    len(etalon_opins),
                    len(list(etalon_opins.iter_sentiment(PositiveLabel()))),
                    len(list(etalon_opins.iter_sentiment(NegativeLabel())))]

        df.loc['sum'] = [float(df[c].sum()) for c in columns]

        df.loc['found'] = None
        df.loc['found']['t_all'] = float(df.loc['sum']['t_all']) / df.loc['sum']['e_all']
        df.loc['found']['t_pos'] = float(df.loc['sum']['t_pos']) / df.loc['sum']['e_pos']
        df.loc['found']['t_neg'] = float(df.loc['sum']['t_neg']) / df.loc['sum']['e_neg']

        return df


# TODO. move from this file.
class FilesToCompare:

    def __init__(self, test_filepath, etalon_filepath, index):
        assert(isinstance(test_filepath, unicode))
        assert(isinstance(etalon_filepath, unicode))
        assert(isinstance(index, int))
        self.test_fp_ = test_filepath
        self.etalon_fp_ = etalon_filepath
        self.index_ = index

    @property
    def test_filepath(self):
        return self.test_fp_

    @property
    def etalon_filepath(self):
        return self.etalon_fp_

    @property
    def index(self):
        return self.index_
