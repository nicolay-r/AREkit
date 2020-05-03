from os import path

import pandas as pd

import io_utils
from arekit.common.experiment.base import BaseExperiment
from arekit.contrib.bert.formatters.opinions.base import BertOpinionsFormatter
from arekit.contrib.bert.formatters.row_ids.base import BaseIDFormatter
from arekit.contrib.bert.formatters.utils import generate_filename, get_output_dir


class BertResults(object):

    ID = 'id'

    def __init__(self, ids_formatter):
        assert(isinstance(ids_formatter, BaseIDFormatter))
        self.__ids_formatter = ids_formatter
        self.__df = None

    @property
    def _IdsFormatter(self):
        return self.__ids_formatter

    # region public methods

    def from_tsv(self, data_type, experiment, ids_values):
        assert(isinstance(data_type, unicode))
        assert(isinstance(experiment, BaseExperiment))

        filepath = self.__get_filepath(data_type=data_type,
                                       experiment=experiment)

        self.__df = pd.read_csv(filepath, sep='\t', header=None)

        assert(len(ids_values) == len(self.__df))
        self.__df[self.ID] = self._get_column_header()

    def iter_wrapped_linked_text_opinions(self, news_id, bert_opinions):
        assert(isinstance(bert_opinions, BertOpinionsFormatter))
        raise NotImplementedError()

    def iter_news_ids(self):
        sample_row_ids = self.__df[self.ID].tolist()
        all_news = [self.__ids_formatter.parse_news_in_sample_id(row_id) for row_id in sample_row_ids]
        for news_id in set(all_news):
            yield news_id

    # endregion

    # region protected methods

    def _get_column_header(self):
        raise NotImplementedError()

    def _iter_linked_opinions_df(self, news_id):
        news_id_pattern = self.__ids_formatter.create_news_id_pattern(news_id=news_id)
        news_samples_df = self.__df[self.__df[self.ID].str.contains(news_id_pattern)]

        opinion_ids = [self.__ids_formatter.parse_opinion_in_opinion_id(opinion_id)
                       for opinion_id in news_samples_df[self.ID]]

        for opinion_id in opinion_ids:
            opin_id_pattern = self.__ids_formatter.create_opinion_id_pattern(opinion_id)
            linked_opins_df = news_samples_df[news_samples_df[self.ID].str.contains(opin_id_pattern)]
            yield linked_opins_df

    # endregion

    # region private methods

    @staticmethod
    def __get_filepath(data_type, experiment):
        assert(isinstance(experiment, BaseExperiment))

        fname = generate_filename(data_type=data_type,
                                  experiment=experiment,
                                  prefix=u'samples')

        filepath = path.join(get_output_dir(experiment=experiment), fname)

        io_utils.create_dir_if_not_exists(filepath)

        return filepath

    # endregion

    def __len__(self):
        return len(self.__df)
