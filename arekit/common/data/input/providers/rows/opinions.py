from collections import OrderedDict

from arekit.common.data import const
from arekit.common.data.input.providers.rows.base import BaseRowProvider
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.dataset.text_opinions.enums import EntityEndType
from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.parsed.base import ParsedNews


class BaseOpinionsRowProvider(BaseRowProvider):

    @staticmethod
    def __create_opinion_row(parsed_news, text_opinions_linkage):
        """
        row format: [id, src, target, label]
        """
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(text_opinions_linkage, TextOpinionsLinkage))

        row = OrderedDict()

        src_value = TextOpinionHelper.extract_entity_value(
            parsed_news=parsed_news,
            text_opinion=text_opinions_linkage.First,
            end_type=EntityEndType.Source)

        target_value = TextOpinionHelper.extract_entity_value(
            parsed_news=parsed_news,
            text_opinion=text_opinions_linkage.First,
            end_type=EntityEndType.Target)

        row[const.ID] = MultipleIDProvider.create_opinion_id(
            text_opinions_linkage=text_opinions_linkage,
            index_in_linked=0)

        row[const.DOC_ID] = text_opinions_linkage.First.DocID

        row[const.SOURCE] = src_value
        row[const.TARGET] = target_value

        return row

    def _provide_rows(self, parsed_news, text_opinion_linkage, idle_mode):
        if idle_mode:
            yield None
        else:
            yield BaseOpinionsRowProvider.__create_opinion_row(parsed_news=parsed_news,
                                                               text_opinions_linkage=text_opinion_linkage)
