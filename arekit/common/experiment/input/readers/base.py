import pandas as pd


class BaseInputReader(object):

    def __init__(self, df):
        assert(isinstance(df, pd.DataFrame))
        self._df = df

    def rows_count(self):
        return len(self._df)
