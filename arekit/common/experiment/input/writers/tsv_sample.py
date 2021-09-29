import logging

from arekit.common.experiment import const
from arekit.common.experiment.input.providers.columns.base import BaseColumnsProvider
from arekit.common.experiment.input.storages.base import BaseRowsStorage
from arekit.common.experiment.input.writers.base import BaseWriter
from arekit.common.utils import create_dir_if_not_exists

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TsvSampleWriter(BaseWriter):

    def __init__(self, balance, write_header):
        assert(isinstance(balance, bool))
        super(TsvSampleWriter, self).__init__()
        self.__balance = balance
        self.__write_header = write_header

    def save(self, storage, target):
        assert(isinstance(storage, BaseRowsStorage))
        assert(isinstance(target, str))

        create_dir_if_not_exists(target)

        # Temporary hack, remove it in future.
        df = storage.DataFrame

        if self.__balance:
            logger.info("Start balancing...")
            df._balance(const.LABEL)
            logger.info("Balancing completed!")

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
