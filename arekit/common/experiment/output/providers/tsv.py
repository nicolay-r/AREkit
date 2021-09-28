import pandas as pd
from arekit.common.experiment.output.providers.base import BaseOutputProvider


# TODO. 206. Rename as reader.
class TsvBaseOutputProvider(BaseOutputProvider):

    def __init__(self, has_output_header):
        assert(isinstance(has_output_header, bool))
        self.__has_output_header = has_output_header
        # TODO. 206. Remove (this is a part of the storage)
        self.__df = None

    # region private methods

    def _csv_to_dataframe(self, filepath):
        return pd.read_csv(filepath,
                           sep='\t',
                           index_col=False,
                           header='infer' if self.__has_output_header else None,
                           encoding='utf-8')

    # endregion

    # TODO. 206. Remove (this is a part of the storage)
    @property
    def DataFrame(self):
        return self.__df

    def load(self, source):
        assert (isinstance(source, str))
        self.__df = self._csv_to_dataframe(filepath=source)
