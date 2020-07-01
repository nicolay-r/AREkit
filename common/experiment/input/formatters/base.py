import logging
from os import path

import numpy as np
import pandas as pd

import io_utils
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.providers.opinions import OpinionProvider

logger = logging.getLogger(__name__)


class BaseBertRowsFormatter(object):

    ROW_ID = 'row_id'

    def __init__(self, data_type):
        assert(isinstance(data_type, DataType))
        self._data_type = data_type
        self._df = self.__create_empty_df()

    @staticmethod
    def formatter_type_log_name():
        return u""

    # region protected methods

    def _get_columns_list_with_types(self):
        dtypes_list = []
        dtypes_list.append((BaseBertRowsFormatter.ROW_ID, 'int32'))
        return dtypes_list

    @staticmethod
    def _iter_by_rows(opinion_provider):
        raise NotImplementedError()

    def __create_empty_df(self):
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

        rows_count = sum(1 for _ in self._iter_by_rows(opinion_provider))

        p_prev = 0
        self.__fill_with_blank_rows(rows_count)
        for row_index, row in enumerate(self._iter_by_rows(opinion_provider)):
            for column, value in row.iteritems():
                self.__set_value(row_ind=row_index,
                                 column=column,
                                 value=value)

            current_work = row_index + 1
            total_work = rows_count
            percent = round(100 * float(current_work) / total_work, 2)

            if percent - p_prev > 5 or (current_work == total_work):
                logging.info("{f_type} ('{d_type}') added: {c}/{t} ({p}%)".format(
                    f_type=self.formatter_type_log_name(),
                    d_type=self._data_type,
                    c=current_work,
                    t=total_work,
                    p=percent))
                p_prev = percent

    def get_filepath(self, data_type, experiment):
        return self.get_filepath_static(data_type=data_type,
                                        experiment=experiment,
                                        prefix=self.formatter_type_log_name())

    @staticmethod
    def get_filepath_static(data_type, experiment, prefix):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(data_type, DataType))
        assert(isinstance(prefix, unicode))

        filename = BaseBertRowsFormatter.__generate_filename(data_type=data_type,
                                                             experiment=experiment,
                                                             prefix=prefix)

        out_dir = BaseBertRowsFormatter.__get_output_dir(experiment=experiment)

        filepath = path.join(out_dir, filename)
        io_utils.create_dir_if_not_exists(filepath)

        return filepath

    @staticmethod
    def __generate_filename(data_type, experiment, prefix):
        assert(isinstance(data_type, DataType))
        assert(isinstance(prefix, unicode))
        assert(isinstance(experiment, BaseExperiment))

        return u"{prefix}-{data_type}-{cv_index}.tsv.gz".format(
            prefix=prefix,
            data_type=data_type,
            cv_index=experiment.DataIO.CVFoldingAlgorithm.IterationIndex)

    @staticmethod
    def __get_output_dir(experiment):
        assert(isinstance(experiment, BaseExperiment))
        return experiment.DataIO.get_model_root()
