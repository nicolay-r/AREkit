import logging
from os import path

import numpy as np
import pandas as pd
from tqdm import tqdm

import io_utils
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.contrib.bert.providers.opinions import OpinionProvider


logger = logging.getLogger(__name__)


class BaseBertRowsFormatter(object):

    ROW_ID = 'row_id'

    def __init__(self, data_type):
        assert(isinstance(data_type, unicode))
        self._data_type = data_type
        self._df = self._create_empty_df()

    @staticmethod
    def formatter_type_log_name():
        return u""

    # region protected methods

    def _get_columns_list_with_types(self):
        dtypes_list = []
        dtypes_list.append((BaseBertRowsFormatter.ROW_ID, 'int32'))
        return dtypes_list

    @staticmethod
    def _iter_by_rows(opinion_provider, idle_mode):
        raise NotImplementedError()

    def _create_empty_df(self):
        data = np.empty(0, dtype=np.dtype(self._get_columns_list_with_types()))
        return pd.DataFrame(data)

    # endregion

    # region private methods

    @staticmethod
    def _set_value(df, row_ind, column, value):
        df.at[row_ind, column] = value

    def __fill_with_blank_rows(self, rows_count):
        assert(isinstance(rows_count, int))
        self._fast_init_df(df=self._df,
                           rows_count=rows_count)

    @staticmethod
    def _fast_init_df(df, rows_count):
        df[BaseBertRowsFormatter.ROW_ID] = range(rows_count)
        df.set_index(BaseBertRowsFormatter.ROW_ID, inplace=True)

    # endregion

    def format(self, opinion_provider):
        assert(isinstance(opinion_provider, OpinionProvider))

        rows_count = sum(1 for _ in self._iter_by_rows(opinion_provider, idle_mode=True))

        self.__fill_with_blank_rows(rows_count)

        it = tqdm(iterable=self._iter_by_rows(opinion_provider, idle_mode=False),
                  total=rows_count,
                  desc="{fmt}-[{dtype}]".format(fmt=self.formatter_type_log_name(),
                                                dtype=self._data_type),
                  miniters=rows_count / 200,
                  ncols=120)

        for row_index, row in enumerate(it):
            for column, value in row.iteritems():
                self._set_value(df=self._df, row_ind=row_index, column=column, value=value)

    def get_filepath(self, data_type, experiment, is_results=False):
        assert(isinstance(is_results, bool))

        return self.get_filepath_static(data_type=data_type,
                                        experiment=experiment,
                                        prefix=self.formatter_type_log_name(),
                                        is_results=is_results,
                                        is_csv=False,
                                        zipped=True)

    @staticmethod
    def get_filepath_static(data_type, experiment, prefix, is_csv, zipped, is_results):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(data_type, unicode))
        assert(isinstance(prefix, unicode))
        assert(isinstance(is_csv, bool))
        assert(isinstance(zipped, bool))
        assert(isinstance(is_results, bool))

        filename = BaseBertRowsFormatter.__generate_filename(
            data_type=data_type,
            prefix=prefix,
            cv_index=experiment.DataIO.CVFoldingAlgorithm.IterationIndex,
            is_csv=is_csv,
            zipped=zipped)

        out_dir = experiment.DataIO.get_model_results_root() if is_results else \
            BaseBertRowsFormatter.__get_output_dir(experiment=experiment)

        filepath = path.join(out_dir, filename)
        io_utils.create_dir_if_not_exists(filepath)

        return filepath

    @staticmethod
    def __generate_filename(data_type, cv_index, prefix, is_csv, zipped):
        assert(isinstance(data_type, unicode))
        assert(isinstance(cv_index, int))
        assert(isinstance(prefix, unicode))
        assert(isinstance(is_csv, bool))
        assert(isinstance(zipped, bool))

        return u"{prefix}-{data_type}-{cv_index}.{is_csv}{zipped_optionally}".format(
            prefix=prefix,
            data_type=data_type,
            cv_index=cv_index,
            is_csv=u'csv' if is_csv else u'tsv',
            zipped_optionally=u'.gz' if zipped else u'')

    @staticmethod
    # TODO. Add results flag
    def __get_output_dir(experiment):
        assert(isinstance(experiment, BaseExperiment))
        # TODO. Use output_root for reslts
        return experiment.DataIO.get_model_root()
