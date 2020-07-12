import pandas as pd

from arekit.common.experiment import const
from arekit.common.experiment.input.readers.base import BaseInputReader


class InputOpinionReader(BaseInputReader):

    def __init__(self, df):
        super(InputOpinionReader, self).__init__(df)

    @classmethod
    def from_tsv(cls, filepath):
        df = pd.read_csv(filepath,
                         sep='\t',
                         header=None,
                         compression='gzip',
                         names=[const.ID, const.SOURCE, const.TARGET])

        return cls(df)

    def provide_opinion_info_by_opinion_id(self, opinion_id):
        assert(isinstance(opinion_id, unicode))

        opinion_row = self._df[self._df[const.ID] == opinion_id]
        df_row = opinion_row.iloc[0].tolist()

        news_id = df_row[0]
        source = df_row[1].decode('utf-8')
        target = df_row[2].decode('utf-8')

        return news_id, source, target


