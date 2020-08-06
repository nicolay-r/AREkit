import gc
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.providers.opinions import OpinionProvider

logger = logging.getLogger(__name__)


class BaseRowsFormatter(object):

    ROW_ID = 'row_id'

    def __init__(self, data_type):
        assert(isinstance(data_type, DataType))
        self._data_type = data_type
        self._df = self._create_empty_df()

    @staticmethod
    def formatter_type_log_name():
        return u""

    # region protected methods

    def _get_columns_list_with_types(self):
        dtypes_list = []
        dtypes_list.append((BaseRowsFormatter.ROW_ID, 'int32'))
        return dtypes_list

    @staticmethod
    def _iter_by_rows(opinion_provider, idle_mode):
        raise NotImplementedError()

    def _create_empty_df(self):
        data = np.empty(0, dtype=np.dtype(self._get_columns_list_with_types()))
        return pd.DataFrame(data)

    # endregion

    # region private methods

    def __set_value(self, row_ind, column, value):
        self._df.at[row_ind, column] = value

    def __fill_with_blank_rows(self, rows_count):
        assert(isinstance(rows_count, int))
        self._df[self.ROW_ID] = range(rows_count)
        self._df.set_index(self.ROW_ID, inplace=True)

    # endregion

    def format(self, opinion_provider):
        assert(isinstance(opinion_provider, OpinionProvider))

        logger.info(u"Calculating rows count ... ")
        rows_count = sum(1 for _ in self._iter_by_rows(opinion_provider, idle_mode=True))
        logger.info(u"Completed!")

        logger.info(u"Filling with blank rows: {}".format(rows_count))
        self.__fill_with_blank_rows(rows_count)
        logger.info(u"Completed!")

        desc = u"{fmt}-{dtype}".format(fmt=self.formatter_type_log_name(),
                                       dtype=self._data_type)

        iter = tqdm(iterable=self._iter_by_rows(opinion_provider, idle_mode=False),
                    desc=desc,
                    total=rows_count,
                    ncols=80)

        for row_index, row in enumerate(iter):
            for column, value in row.iteritems():
                self.__set_value(row_ind=row_index,
                                 column=column,
                                 value=value)

    def __iter__(self):
        for row in self._df.iterrows():
            yield row

    def dispose_dataframe(self):
        del self._df
        gc.collect()