import pandas as pd
from core.evaluation.labels import Label, PositiveLabel, NegativeLabel
from core.source.opinion import OpinionCollection
from core.source.synonyms import SynonymsCollection


# TODO: Move into stat branch of erc.
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
            if etalon_opins.has_synonymous_opinion(opin, sentiment):
                founded += 1

        return founded

    @staticmethod
    def get_method_statistic(files_to_compare_list, synonyms):
        """
            Calculate statistic based on result files
            files_to_compare_list: list
                list of FilesToCompare objects
            stemmer: Stemmer
        """
        assert(isinstance(synonyms, SynonymsCollection))

        columns = ["t_all", "t_pos", "t_neg", "e_all", "e_pos", "e_neg"]

        df = pd.DataFrame(columns=columns)
        for files_to_compare in files_to_compare_list:

            assert(isinstance(files_to_compare, FilesToCompare))
            test_opins = OpinionCollection.from_file(files_to_compare.TestFilepath, synonyms)
            etalon_opins = OpinionCollection.from_file(files_to_compare.EtalonFilepath, synonyms)

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


# TODO. move from this file. (cmp.py)
class FilesToCompare:

    def __init__(self, test_filepath, etalon_filepath, index):
        assert(isinstance(test_filepath, unicode))
        assert(isinstance(etalon_filepath, unicode))
        assert(isinstance(index, int))
        self.__test_fp = test_filepath
        self.__etalon_fp = etalon_filepath
        self.__index = index

    @property
    def TestFilepath(self):
        return self.__test_fp

    @property
    def EtalonFilepath(self):
        return self.__etalon_fp

    @property
    def index(self):
        return self.__index
