from os import path

import pandas as pd

import io_utils
from arekit.common.experiment.base import BaseExperiment
from arekit.contrib.bert.formatters.utils import generate_filename, get_output_dir


class BertResults(object):

    def __init__(self, df):
        assert(isinstance(df, pd.DataFrame))
        self.__df = df

    @classmethod
    def from_tsv(cls, data_type, experiment):
        assert(isinstance(data_type, unicode))
        assert(isinstance(experiment, BaseExperiment))

        filepath = cls.__get_filepath(data_type=data_type,
                                      experiment=experiment)

        df = pd.read_csv(filepath, sep='\t', header=None)

        cls(df)

    @staticmethod
    def __get_filepath(data_type, experiment):
        assert(isinstance(experiment, BaseExperiment))

        fname = generate_filename(data_type=data_type,
                                  experiment=experiment,
                                  prefix=u'samples')

        filepath = path.join(get_output_dir(experiment=experiment), fname)

        io_utils.create_dir_if_not_exists(filepath)

        return filepath

    def __len__(self):
        return len(self.__df)
