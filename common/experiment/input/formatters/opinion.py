from collections import OrderedDict
from arekit.common.experiment import const
from arekit.common.experiment.input.formatters.base_row import BaseRowsFormatter
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.dataset.text_opinions.enums import EntityEndType


class BaseOpinionsFormatter(BaseRowsFormatter):

    # region methods

    def __init__(self, data_type):
        super(BaseOpinionsFormatter, self).__init__(data_type=data_type)

    @staticmethod
    def formatter_type_log_name():
        return u"opinion"

    def _get_columns_list_with_types(self):
        dtypes_list = super(BaseOpinionsFormatter, self)._get_columns_list_with_types()
        dtypes_list.append((const.ID, unicode))
        dtypes_list.append((const.SOURCE, unicode))
        dtypes_list.append((const.TARGET, unicode))
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

        row[const.ID] = MultipleIDProvider.create_opinion_id(
            opinion_provider=opinion_provider,
            linked_opinions=linked_wrapper,
            index_in_linked=0)
        row[const.SOURCE] = src_value
        row[const.TARGET] = target_value

        return row

    @staticmethod
    def _iter_by_rows(opinion_provider):
        assert(isinstance(opinion_provider, OpinionProvider))

        linked_iter = opinion_provider.iter_linked_opinion_wrappers(balance=False,
                                                                    supported_labels=None)

        for linked_wrapper in linked_iter:
            yield BaseOpinionsFormatter.__create_opinion_row(
                opinion_provider=opinion_provider,
                linked_wrapper=linked_wrapper)

    # endregion

    def save(self, filepath):
        self._df.to_csv(filepath,
                        sep='\t',
                        encoding='utf-8',
                        columns=[c for c in self._df.columns if c != self.ROW_ID],
                        index=False,
                        compression='gzip',
                        header=False)

