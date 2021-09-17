import logging

from arekit.common.experiment import const
from arekit.common.experiment.input.formatters.helper.balancing import SampleRowBalancerHelper
from arekit.common.experiment.input.storages.sample import BaseSampleStorage
from arekit.common.utils import create_dir_if_not_exists

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TsvSampleStorage(BaseSampleStorage):

    def __init__(self, filepath, store_labels, balance, write_header):
        assert(isinstance(filepath, str))
        assert(isinstance(balance, bool))
        super(TsvSampleStorage, self).__init__(store_labels)

        self.__balance = balance
        self.__filepath = filepath
        self.__write_header = write_header

    def save(self):

        create_dir_if_not_exists(self.__filepath)

        if self.__balance:
            logger.info("Start balancing...")
            balanced_df = SampleRowBalancerHelper.calculate_balanced_df(
                df=self._df,
                create_blank_df=lambda size: self._create_blank_df(size),
                output_labels_uint=self._output_labels_uint)
            logger.info("Balancing completed!")
            self._dispose_dataframe()
            self._df = balanced_df

        logger.info("Saving... {shape}: {filepath}".format(
            shape=self._df.shape,  # self._df.shape,
            filepath=self.__filepath))
        self._df.sort_values(by=[const.ID], ascending=True)
        self._df.to_csv(self.__filepath,
                        sep='\t',
                        encoding='utf-8',
                        columns=[c for c in self._df.columns if c != self.ROW_ID],
                        index=False,
                        float_format="%.0f",
                        compression='gzip',
                        header=self.__write_header)
        logger.info("Saving completed!")
        logger.info(self._df.info())
