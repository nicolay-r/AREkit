import pandas as pd

from arekit.common.experiment.input.formatters.base_row import BaseRowsFormatter
from arekit.common.experiment.input.providers.row_ids.base import BaseIDProvider
from arekit.common.experiment.input.readers.opinion import InputOpinionReader
from arekit.common.linked.opinions.wrapper import LinkedOpinionWrapper
from arekit.common.opinions.base import Opinion


class BaseOutput(object):

    ID = 'id'

    def __init__(self, ids_formatter):
        assert(isinstance(ids_formatter, BaseIDProvider))
        self.__ids_formatter = ids_formatter
        self.__df = None

    @property
    def _IdsFormatter(self):
        return self.__ids_formatter

    # region public methods

    def init_from_tsv(self, source_dir, filename_template, ids_values):
        assert(isinstance(source_dir, unicode))
        assert(isinstance(filename_template, unicode))

        filepath = BaseRowsFormatter.get_filepath_static(out_dir=source_dir,
                                                         template=filename_template,
                                                         prefix=u"result")

        self.__df = pd.read_csv(filepath,
                                sep='\t',
                                index_col=False,
                                header=None)
        self.__df.columns = self._get_column_header()
        self.__df.insert(0, self.ID, ids_values)
        self.__df.set_index(self.ID)

    def iter_news_ids(self):
        sample_row_ids = self.__df[self.ID].tolist()
        all_news = [self.__ids_formatter.parse_news_in_sample_id(sample_id) for sample_id in sample_row_ids]
        for news_id in set(all_news):
            yield news_id

    def iter_linked_opinions(self, news_id, opinions_reader):
        assert(isinstance(news_id, int))
        assert(isinstance(opinions_reader, InputOpinionReader))

        for linked_df in self.__iter_linked_opinions_df(news_id=news_id):
            assert(isinstance(linked_df, pd.DataFrame))

            opinions_iter = self._iter_by_opinions(linked_df=linked_df,
                                                   opinions_reader=opinions_reader)

            yield LinkedOpinionWrapper(linked_data=opinions_iter)

    # endregion

    # region protected methods

    def _get_column_header(self):
        raise NotImplementedError()

    def _iter_by_opinions(self, linked_df, opinions_reader):
        raise NotImplementedError()

    def __iter_linked_opinions_df(self, news_id):
        news_id_pattern = self.__ids_formatter.create_pattern(id_value=news_id,
                                                              p_type=BaseIDProvider.NEWS)
        n_df = self.__df[self.__df[self.ID].str.contains(news_id_pattern)]

        opinion_ids = [self.__ids_formatter.parse_opinion_in_opinion_id(opinion_id)
                       for opinion_id in n_df[self.ID]]

        for opinion_id in set(opinion_ids):
            opin_id_pattern = self.__ids_formatter.create_pattern(id_value=opinion_id,
                                                                  p_type=BaseIDProvider.OPINION)
            linked_opins_df = n_df[n_df[self.ID].str.contains(opin_id_pattern)]
            yield linked_opins_df

    def _compose_opinion_by_opinion_id(self, sample_id, opinions_reader, calc_label_func):
        assert(isinstance(sample_id, unicode))
        assert(isinstance(opinions_reader, InputOpinionReader))
        assert(callable(calc_label_func))

        opinion_id = self.__ids_formatter.convert_sample_id_to_opinion_id(sample_id=sample_id)
        _, source, target = opinions_reader.provide_opinion_info_by_opinion_id(opinion_id=opinion_id)

        return Opinion(source_value=source,
                       target_value=target,
                       sentiment=calc_label_func())

    # endregion

    def __len__(self):
        return len(self.__df.index)
