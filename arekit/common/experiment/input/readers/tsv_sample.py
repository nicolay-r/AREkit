import pandas as pd

from arekit.common.experiment.input.readers.base_sample import BaseInputSampleReader
from arekit.common.experiment.row_ids.base import BaseIDProvider


class TsvInputSampleReader(BaseInputSampleReader):

    @classmethod
    def from_tsv(cls, filepath, row_ids_provider):
        assert(isinstance(filepath, str))
        assert(isinstance(row_ids_provider, BaseIDProvider))

        df = pd.read_csv(filepath,
                         compression='gzip',
                         sep='\t',
                         encoding='utf-8')

        return cls(df=df, row_ids_provider=row_ids_provider)
