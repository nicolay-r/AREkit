from arekit.common.experiment import const
from arekit.common.experiment.input.storages.opinion import BaseOpinionsStorage, logger
from arekit.common.utils import create_dir_if_not_exists


class TsvOpinionsStorage(BaseOpinionsStorage):

    def __init__(self, filepath):
        super(TsvOpinionsStorage, self).__init__()
        self.__filepath = filepath

    def save(self):
        logger.info("Saving... : {}".format(self.__filepath))

        create_dir_if_not_exists(self.__filepath)

        self._df.sort_values(by=[const.ID], ascending=True)
        self._df.to_csv(self.__filepath,
                        sep='\t',
                        encoding='utf-8',
                        columns=[c for c in self._df.columns if c != self.ROW_ID],
                        index=False,
                        compression='gzip')
        logger.info("Saving completed!")
