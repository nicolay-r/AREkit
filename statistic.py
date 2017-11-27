
import pandas as pd
from core.source.opinion import OpinionCollection
from core.labels import Label, PositiveLabel, NegativeLabel


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
    def get_method_statistic(method_name, test_root, etalon_root, test_indices, synonyms_filepath):

        columns = ["t_all", "t_pos", "t_neg", "e_all", "e_pos", "e_neg"]

        df = pd.DataFrame(columns=columns)
        for n in test_indices:

            eo_filepath = "{}/art{}.opin.txt".format(etalon_root, n)
            to_filepath = "{}/art{}.opin.txt".format(test_root, n)

            test_opins = OpinionCollection.from_file(to_filepath, synonyms_filepath)
            etalon_opins = OpinionCollection.from_file(eo_filepath, synonyms_filepath)

            df.loc[n] = [MethodStatistic.founded_opins(test_opins, etalon_opins),
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
