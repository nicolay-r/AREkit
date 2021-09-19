import logging
from arekit.common.experiment.input.storages.base import BaseRowsStorage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseSampleStorage(BaseRowsStorage):

    def __init__(self):
        super(BaseSampleStorage, self).__init__()
        self._output_labels_uint = None

    # region private methods

    def __fast_init_df(self, df, rows_count):
        df[self._columns_provider.ROW_ID] = list(range(rows_count))
        df.set_index(self._columns_provider.ROW_ID, inplace=True)

    # endregion

    # region protected methods

    def _create_blank_df(self, size):
        df = self._create_empty()
        self.__fast_init_df(df=df, rows_count=size)
        return df

    # endregion

    # region public methods

    def set_output_labels_uint(self, labels_uint):
        if self._output_labels_uint is not None:
            raise Exception("Output labels already defined!")
        self._output_labels_uint = labels_uint

    def save(self, target):
        """ This might be implemented in nested classes.
            The default, i.e. pandas-based storage is not considered
            to be saved into the particular target.
        """
        pass

    # endregion

    def __len__(self):
        return len(self._df.index)
