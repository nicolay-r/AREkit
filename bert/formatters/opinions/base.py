from collections import OrderedDict

import pandas as pd

from arekit.bert.formatters.base import BaseBertRowsFormatter
from arekit.bert.providers.opinions import OpinionProvider
from arekit.bert.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.text_opinions.enums import EntityEndType


class BertOpinionsFormatter(BaseBertRowsFormatter):

    ID = 'id'
    SOURCE = 'source'
    TARGET = 'target'

    # region methods

    def __init__(self, data_type):
        super(BertOpinionsFormatter, self).__init__(data_type=data_type)

    @staticmethod
    def formatter_type_log_name():
        return u"opinion"

    def _get_columns_list_with_types(self):
        dtypes_list = super(BertOpinionsFormatter, self)._get_columns_list_with_types()
        dtypes_list.append((BertOpinionsFormatter.ID, unicode))
        dtypes_list.append((BertOpinionsFormatter.SOURCE, unicode))
        dtypes_list.append((BertOpinionsFormatter.TARGET, unicode))
        return dtypes_list

    @staticmethod
    def __create_opinion_row(opinion_provider, linked_wrapper):
        """
        row format: [id, src, target, label]
        """
        assert(isinstance(opinion_provider, OpinionProvider))
        assert(isinstance(linked_wrapper, LinkedTextOpinionsWrapper))

        row = OrderedDict()

        src_value = opinion_provider.get_entity_value(
            text_opinion=linked_wrapper.First,
            end_type=EntityEndType.Source)

        target_value = opinion_provider.get_entity_value(
            text_opinion=linked_wrapper.First,
            end_type=EntityEndType.Target)

        row[BertOpinionsFormatter.ID] = MultipleIDProvider.create_opinion_id(
            opinion_provider=opinion_provider,
            linked_opinions=linked_wrapper,
            index_in_linked=0)
        row[BertOpinionsFormatter.SOURCE] = src_value
        row[BertOpinionsFormatter.TARGET] = target_value

        return row

    @staticmethod
    def _iter_by_rows(opinion_provider):
        assert(isinstance(opinion_provider, OpinionProvider))

        linked_iter = opinion_provider.iter_linked_opinion_wrappers(balance=False,
                                                                    supported_labels=None)

        for linked_wrapper in linked_iter:
            yield BertOpinionsFormatter.__create_opinion_row(
                opinion_provider=opinion_provider,
                linked_wrapper=linked_wrapper)

    # endregion

    def provide_opinion_info_by_opinion_id(self, opinion_id):
        assert(isinstance(opinion_id, unicode))

        opinion_row = self._df[self._df[self.ID] == opinion_id]
        df_row = opinion_row.iloc[0].tolist()

        news_id = df_row[0]
        source = df_row[1].decode('utf-8')
        target = df_row[2].decode('utf-8')

        return news_id, source, target

    def to_tsv_by_experiment(self, experiment):
        assert(isinstance(experiment, BaseExperiment))

        filepath = self.get_filepath(data_type=self._data_type,
                                     experiment=experiment)

        self._df.to_csv(filepath,
                        sep='\t',
                        encoding='utf-8',
                        columns=[c for c in self._df.columns if c != self.ROW_ID],
                        index=False,
                        compression='gzip',
                        header=False)

    def from_tsv(self, experiment):
        assert(isinstance(experiment, BaseExperiment))

        filepath = self.get_filepath(data_type=self._data_type,
                                     experiment=experiment)

        self._df = pd.read_csv(filepath,
                               sep='\t',
                               header=None,
                               compression='gzip',
                               names=[self.ID, self.SOURCE, self.TARGET])

    def get_ids(self):
        return self._df.iloc[0].tolist()

