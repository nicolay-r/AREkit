import logging
from arekit.common.experiment.input.storages.base import BaseRowsStorage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseSampleStorage(BaseRowsStorage):

    def __init__(self):
        super(BaseSampleStorage, self).__init__()

    # region private methods

    def __fast_init_df(self, df, rows_count):
        df[self._columns_provider.ROW_ID] = list(range(rows_count))
        df.set_index(self._columns_provider.ROW_ID, inplace=True)

    # endregion

    # region public methods

    # TODO: 202 to BaseSampleWriter.
    def save(self, target):
        """ This might be implemented in nested classes.
            The default, i.e. pandas-based storage is not considered
            to be saved into the particular target.
        """
        pass

    # endregion

    def __len__(self):
        return len(self._df.index)
