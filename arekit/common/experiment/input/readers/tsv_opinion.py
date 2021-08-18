import pandas as pd

from arekit.common.experiment.input.readers.base_opinion import BaseInputOpinionReader


class TsvInputOpinionReader(BaseInputOpinionReader):

    @classmethod
    def from_tsv(cls, filepath, compression='gzip'):
        df = pd.read_csv(filepath,
                         sep='\t',
                         compression=compression)

        return cls(df)
