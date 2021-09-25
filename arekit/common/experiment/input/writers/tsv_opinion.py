import logging

from arekit.common.experiment import const
from arekit.common.experiment.input.providers.columns.base import BaseColumnsProvider
from arekit.common.experiment.input.writers.base import BaseWriter
from arekit.common.utils import create_dir_if_not_exists

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TsvOpinionsWriter(BaseWriter):

    def __init__(self):
        super(TsvOpinionsWriter, self).__init__()

    def save(self, target):
        assert(isinstance(target, str))

        logger.info("Saving... : {}".format(target))

        create_dir_if_not_exists(target)

        # Refactor later.
        df = self._storage._df

        df.sort_values(by=[const.ID], ascending=True)
        df.to_csv(target,
                  sep='\t',
                  encoding='utf-8',
                  columns=[c for c in df.columns if c != BaseColumnsProvider.ROW_ID],
                  index=False,
                  compression='gzip')

        logger.info("Saving completed!")
