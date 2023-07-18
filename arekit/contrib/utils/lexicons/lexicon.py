from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader


class Lexicon(object):

    @property
    def ToneKey(self):
        return 'tone'

    @property
    def TermKey(self):
        return 'term'

    def __init__(self, dataframe):
        self.__lexicon_df = dataframe

    @classmethod
    def load(cls, filepath, separator=','):
        reader = PandasCsvReader(compression=None, sep=separator)
        return cls(reader.read(filepath))

    def get_score(self, lemma):
        assert(type(lemma) == str)
        s = self.__lexicon_df[lemma.encode('utf-8') == self.__lexicon_df[self.TermKey]]
        return s[self.ToneKey].values[0] if len(s) > 0 else 0

    def has_term(self, term):
        assert(type(term) == str)
        s = self.__lexicon_df[term.encode('utf-8') == self.__lexicon_df[self.TermKey]]
        return len(s) > 0

    def __iter__(self):
        for term in self.__lexicon_df[self.TermKey]:
            yield term

    def __contains__(self, item):
        assert(isinstance(item, str))
        result = self.__lexicon_df[self.__lexicon_df[self.TermKey] == item.encode('utf-8')]
        return len(result) > 0


