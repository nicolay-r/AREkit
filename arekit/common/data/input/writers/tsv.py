import logging

from arekit.common.data import const
from arekit.common.data.input.providers.columns.base import BaseColumnsProvider
from arekit.common.data.input.writers.base import BaseWriter
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.utils import create_dir_if_not_exists

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TsvWriter(BaseWriter):

    def __init__(self, write_header):
        super(TsvWriter, self).__init__()
        self.__write_header = write_header

    def save(self, storage, target):
        assert(isinstance(storage, BaseRowsStorage))
        assert(isinstance(target, str))

        create_dir_if_not_exists(target)

        # Temporary hack, remove it in future.
        df = storage.DataFrame

        logger.info("Saving... {shape}: {filepath}".format(shape=df.shape,  # self._df.shape,
                                                           filepath=target))
        df.sort_values(by=[const.ID], ascending=True)
        df.to_csv(target,
                  sep='\t',
                  encoding='utf-8',
                  columns=[c for c in df.columns if c != BaseColumnsProvider.ROW_ID],
                  index=False,
                  float_format="%.0f",
                  compression='gzip',
                  header=self.__write_header)
        logger.info("Saving completed!")
        logger.info(df.info())
