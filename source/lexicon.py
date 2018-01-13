import pandas as pd

class Lexicon:

    def __init__(self, dataframe):
        self.lexicon = dataframe

    @staticmethod
    def from_file(filepath):
        df = pd.read_csv(filepath, sep=',')
        return Lexicon(df)

    def get_score(self, lemma):
        assert(type(lemma) == unicode)
        s = self.lexicon[lemma.encode('utf-8') == self.lexicon['term']]
        return s['tone'].values[0] if len(s) > 0 else 0


class RelationLexicon:

    def __init__(self, dataframe):
        assert(isinstance(dataframe, pd.DataFrame))
        self._check(dataframe)
        self.lexicon = dataframe
        print self.lexicon.index

    @staticmethod
    def from_file(filepath):
        df = pd.read_csv(filepath, sep=',')
        return RelationLexicon(df)

    @staticmethod
    def _check(df):
        for index in df.index:
            relation = df.loc[index][0]
            assert(len(relation.split('<->')) == 2)

    @staticmethod
    def _create_key(l, r):
        assert(type(l) == unicode)
        assert(type(r) == unicode)
        return ('<->'.join([l.encode('utf-8'), r.encode('utf-8')])).decode('utf-8')

    def get_score(self, left, right):
        assert(type(left) == unicode)
        assert(type(right) == unicode)

        lr_key = self._create_key(left, right).encode('utf-8')
        rl_key = self._create_key(right, left).encode('utf-8')

        lr_score = self.lexicon[lr_key == self.lexicon['relation']]
        rl_score = self.lexicon[rl_key == self.lexicon['relation']]

        if len(lr_score) > 0:
            return lr_score['tone'].values[0]
        if len(rl_score) > 0:
            return rl_score['tone'].values[0]

        return None
