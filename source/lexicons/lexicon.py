import pandas as pd


class Lexicon(object):

    def __init__(self, dataframe):
        self.__lexicon = dataframe

    @classmethod
    def from_csv(cls, filepath, separator=','):
        df = pd.read_csv(filepath, sep=separator)
        return cls(df)

    def get_score(self, lemma):
        assert(type(lemma) == unicode)
        s = self.__lexicon[lemma.encode('utf-8') == self.__lexicon['term']]
        return s['tone'].values[0] if len(s) > 0 else 0

    def has_term(self, term):
        assert(type(term) == unicode)
        s = self.__lexicon[term.encode('utf-8') == self.__lexicon['term']]
        return len(s) > 0


