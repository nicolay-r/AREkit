from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader


class RelationLexicon(object):

    def __init__(self, dataframe):
        self.__check(dataframe)
        self.__lexicon = dataframe

    @classmethod
    def load(cls, filepath, separator=','):
        reader = PandasCsvReader(compression=None, sep=separator)
        return cls(reader.read(filepath))

    @staticmethod
    def __check(df):
        for index in df.index:
            relation = df.loc[index][0]
            assert(len(relation.split('<->')) == 2)

    @staticmethod
    def __create_key(l, r):
        assert(type(l) == str)
        assert(type(r) == str)
        return '<->'.join([l, r])

    def get_score(self, left, right):
        assert(type(left) == str)
        assert(type(right) == str)

        lr_key = self.__create_key(left, right)
        rl_key = self.__create_key(right, left)

        lr_score = self.__lexicon[lr_key == self.__lexicon['relation']]
        rl_score = self.__lexicon[rl_key == self.__lexicon['relation']]

        if len(lr_score) > 0:
            return lr_score['tone'].values[0]
        if len(rl_score) > 0:
            return rl_score['tone'].values[0]

        return None
