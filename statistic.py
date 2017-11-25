
import pandas as pd
from core.source.opinion import OpinionCollection
from core.labels import PositiveLabel, NegativeLabel


class MethodStatistic:

    @staticmethod
    def founded_opins(test_opins, etalon_opins, sentiment=None):
        assert(isinstance(test_opins, OpinionCollection))
        assert(isinstance(etalon_opins, OpinionCollection))
        founded = 0
        for e in test_opins:
            founded += 1 if etalon_opins.has_opinion_by_synonyms(e, sentiment) else 0
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

        df.loc['found_all'] = None
        df.loc['found_pos'] = None
        df.loc['found_neg'] = None
        df.loc['found_all'][0] = float(df.loc['sum']['t_all']) / df.loc['sum']['e_all']
        df.loc['found_pos'][1] = float(df.loc['sum']['t_pos']) / df.loc['sum']['e_pos']
        df.loc['found_neg'][2] = float(df.loc['sum']['t_neg']) / df.loc['sum']['e_neg']

        return df
