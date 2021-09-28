from collections import OrderedDict

from arekit.common.dataset.text_opinions.enums import EntityEndType
from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.experiment import const
from arekit.common.experiment.input.providers.rows.base import BaseRowProvider
from arekit.common.experiment.row_ids.multiple import MultipleIDProvider
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.news.parsed.base import ParsedNews


class BaseOpinionsRowProvider(BaseRowProvider):

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
            yield BaseOpinionsRowProvider.__create_opinion_row(parsed_news=parsed_news,
                                                               linked_wrapper=linked_wrapper)
