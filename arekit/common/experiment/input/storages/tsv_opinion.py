from arekit.common.experiment import const
from arekit.common.experiment.input.storages.opinion import BaseOpinionsStorage, logger
from arekit.common.utils import create_dir_if_not_exists


class TsvOpinionsStorage(BaseOpinionsStorage):

    def __init__(self):
        super(TsvOpinionsStorage, self).__init__()

    def save(self, target):
        assert(isinstance(target, str))

        logger.info("Saving... : {}".format(target))

        create_dir_if_not_exists(target)

        self._df.sort_values(by=[const.ID], ascending=True)
        self._df.to_csv(target,
                        sep='\t',
                        encoding='utf-8',
                        columns=[c for c in self._df.columns if c != self._columns_provider.ROW_ID],
                        index=False,
                        compression='gzip')
        logger.info("Saving completed!")
