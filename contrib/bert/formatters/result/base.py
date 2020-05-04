from os import path

import pandas as pd

import io_utils
from arekit.common.experiment.base import BaseExperiment
from arekit.common.linked_text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.opinions.base import Opinion
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

    def iter_news_ids(self):
        sample_row_ids = self.__df[self.ID].tolist()
        all_news = [self.__ids_formatter.parse_news_in_sample_id(row_id) for row_id in sample_row_ids]
        for news_id in set(all_news):
            yield news_id

    def iter_wrapped_linked_text_opinions(self, news_id, bert_opinions):
        assert(isinstance(news_id, int))
        assert(isinstance(bert_opinions, BertOpinionsFormatter))

        for linked_df in self._iter_linked_opinions_df(news_id=news_id):
            assert(isinstance(linked_df, pd.DataFrame))
            opinions = list(self._to_opinions(linked_df=linked_df, bert_opinions=bert_opinions))
            yield LinkedTextOpinionsWrapper(linked_text_opinions=opinions)

    # endregion

    # region protected methods

    def _get_column_header(self):
        raise NotImplementedError()

    def _to_opinions(self, linked_df, bert_opinions):
        """
        lined_df: DataFrame
            describes a linked opinions in dataframe (related rows)
        bert_opinions: BertOpinions
        """
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

    def _compose_opinion_by_opinion_id(self, sample_id, bert_opinions, calc_label_func):
        assert(isinstance(sample_id, unicode))
        assert(isinstance(bert_opinions, BertOpinionsFormatter))
        assert(callable(calc_label_func))

        opinion_id = self.__ids_formatter.convert_sample_id_to_opinion_id(sample_id=sample_id)
        _, source, target = bert_opinions.provide_opinion_info_by_opinion_id(opinion_id=opinion_id)

        return Opinion(source_value=source,
                       target_value=target,
                       sentiment=calc_label_func())

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
