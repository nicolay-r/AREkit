from collections import OrderedDict

from arekit.common.data import const
from arekit.common.data.input.providers.rows.base import BaseRowProvider
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.parsed.providers.entity_service import EntityEndType, EntityServiceProvider


class BaseOpinionsRowProvider(BaseRowProvider):

    @staticmethod
    def __create_opinion_row(entity_service, text_opinions_linkage):
        """
        row format: [id, src, target, label]
        """
        assert(isinstance(entity_service, EntityServiceProvider))
        assert(isinstance(text_opinions_linkage, TextOpinionsLinkage))

        row = OrderedDict()

        src_value = entity_service.extract_entity_value(
            text_opinion=text_opinions_linkage.First,
            end_type=EntityEndType.Source)

        target_value = entity_service.extract_entity_value(
            text_opinion=text_opinions_linkage.First,
            end_type=EntityEndType.Target)

        row[const.ID] = MultipleIDProvider.create_opinion_id(
            text_opinions_linkage=text_opinions_linkage,
            index_in_linked=0)

        row[const.DOC_ID] = text_opinions_linkage.First.DocID

        row[const.SOURCE] = src_value
        row[const.TARGET] = target_value

        return row

    def _provide_rows(self, parsed_news, entity_service, text_opinion_linkage, idle_mode):
        if idle_mode:
            yield None
        else:
            yield BaseOpinionsRowProvider.__create_opinion_row(entity_service=entity_service,
                                                               text_opinions_linkage=text_opinion_linkage)
