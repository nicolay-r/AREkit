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
