import logging
from collections import OrderedDict

from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.experiment import const
from arekit.common.experiment.input.formatters.base_row import BaseRowsFormatter
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.dataset.text_opinions.enums import EntityEndType
from arekit.common.news.parsed.base import ParsedNews

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseOpinionsFormatter(BaseRowsFormatter):

    # region methods

    def __init__(self, data_type):
        super(BaseOpinionsFormatter, self).__init__(data_type=data_type)

    def _get_columns_list_with_types(self):
        dtypes_list = super(BaseOpinionsFormatter, self)._get_columns_list_with_types()
        dtypes_list.append((const.ID, str))
        dtypes_list.append((const.NEWS_ID, 'int32'))
        dtypes_list.append((const.SOURCE, str))
        dtypes_list.append((const.TARGET, str))
        return dtypes_list

    @staticmethod
    def __create_opinion_row(parsed_news, linked_wrapper):
        """
        row format: [id, src, target, label]
        """
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(linked_wrapper, LinkedTextOpinionsWrapper))

        row = OrderedDict()

        src_value = TextOpinionHelper.extract_entity_value(
            parsed_news=parsed_news,
            text_opinion=linked_wrapper.First,
            end_type=EntityEndType.Source)

        target_value = TextOpinionHelper.extract_entity_value(
            parsed_news=parsed_news,
            text_opinion=linked_wrapper.First,
            end_type=EntityEndType.Target)

        row[const.ID] = MultipleIDProvider.create_opinion_id(
            linked_opinions=linked_wrapper,
            index_in_linked=0)

        row[const.NEWS_ID] = linked_wrapper.First.NewsID

        row[const.SOURCE] = src_value
        row[const.TARGET] = target_value

        return row

    def _provide_rows(self, parsed_news, linked_wrapper, idle_mode):
        if idle_mode:
            yield None
        else:
            yield BaseOpinionsFormatter.__create_opinion_row(parsed_news=parsed_news,
                                                             linked_wrapper=linked_wrapper)

    # endregion

    def save(self, filepath):
        logger.info("Saving... : {}".format(filepath))
        self._df.sort_values(by=[const.ID], ascending=True)
        self._df.to_csv(filepath,
                        sep='\t',
                        encoding='utf-8',
                        columns=[c for c in self._df.columns if c != self.ROW_ID],
                        index=False,
                        compression='gzip')
        logger.info("Saving completed!")

