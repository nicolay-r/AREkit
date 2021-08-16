import pandas as pd


class RelationLexicon(object):

    def __init__(self, df):
        assert(isinstance(df, pd.DataFrame))
        self.__check(df)
        self.__lexicon = df

    @classmethod
    def from_csv(cls, filepath, separator=','):
        df = pd.read_csv(filepath, sep=separator)
        return cls(df)

    @staticmethod
    def __check(df):
        for index in df.index:
            relation = df.loc[index][0]
            assert(len(relation.split('<->')) == 2)

    @staticmethod
    def __create_key(l, r):
        assert(type(l) == str)
        assert(type(r) == str)
        return ('<->'.join([l.encode('utf-8'), r.encode('utf-8')])).decode('utf-8')

    def get_score(self, left, right):
        assert(type(left) == str)
        assert(type(right) == str)

        lr_key = self.__create_key(left, right).encode('utf-8')
        rl_key = self.__create_key(right, left).encode('utf-8')

        lr_score = self.__lexicon[lr_key == self.__lexicon['relation']]
        rl_score = self.__lexicon[rl_key == self.__lexicon['relation']]

        if len(lr_score) > 0:
            return lr_score['tone'].values[0]
        if len(rl_score) > 0:
            return rl_score['tone'].values[0]

        return None