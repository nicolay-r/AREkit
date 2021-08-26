import gc
import logging

import numpy as np
import pandas as pd

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.utils import progress_bar_defined, progress_bar_iter

logger = logging.getLogger(__name__)


class BaseRowsFormatter(object):

    ROW_ID = 'row_id'

    def __init__(self, data_type):
        assert(isinstance(data_type, DataType))
        self._data_type = data_type
        self._df = self._create_empty_df()

    # region protected methods

    def _get_columns_list_with_types(self):
        dtypes_list = list()
        dtypes_list.append((BaseRowsFormatter.ROW_ID, 'int32'))
        return dtypes_list

    def _iter_by_rows(self, opinion_provider, idle_mode):
        assert(isinstance(opinion_provider, OpinionProvider))

        for parsed_news, linked_wrapper in opinion_provider.iter_linked_opinion_wrappers():

            rows_it = self._provide_rows(parsed_news=parsed_news,
                                         linked_wrapper=linked_wrapper,
                                         idle_mode=idle_mode)

            for row in rows_it:
                yield row

    def _provide_rows(self, parsed_news, linked_wrapper, idle_mode):
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
        self._df[self.ROW_ID] = list(range(rows_count))
        self._df.set_index(self.ROW_ID, inplace=True)

    # endregion

    def format(self, opinion_provider, desc=""):
        assert(isinstance(opinion_provider, OpinionProvider))

        logged_rows_it = progress_bar_iter(self._iter_by_rows(opinion_provider, idle_mode=True),
                                           desc="Calculating rows count",
                                           unit="rows")
        rows_count = sum(1 for _ in logged_rows_it)

        logger.info("Filling with blank rows: {}".format(rows_count))
        self.__fill_with_blank_rows(rows_count)
        logger.info("Completed!")

        it = progress_bar_defined(iterable=self._iter_by_rows(opinion_provider, idle_mode=False),
                                  desc="{fmt}-{dtype}".format(fmt=desc, dtype=self._data_type),
                                  total=rows_count)

        for row_index, row in enumerate(it):
            for column, value in row.items():
                self.__set_value(row_ind=row_index,
                                 column=column,
                                 value=value)

        logger.info(self._df.info())

    def __iter__(self):
        for row in self._df.iterrows():
            yield row

    def dispose_dataframe(self):
        del self._df
        gc.collect()
