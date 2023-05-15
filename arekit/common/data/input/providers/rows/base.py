import collections
import logging

from arekit.common.data.input.providers.contents import ContentsProvider
from arekit.common.linkage.base import LinkedDataWrapper
from arekit.common.news.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.news.parsed.service import ParsedNewsService

logger = logging.getLogger(__name__)


class BaseRowProvider(object):
    """ Base provider for rows that suppose to be filled into BaseRowsStorage.
    """

    # region protected methods

    # TODO. This might be also generalized.
    # TODO. Idle-mode is also a implementation and task specific parameter, i.e. might be removed from here.
    def _provide_rows(self, parsed_news, entity_service, text_opinion_linkage, idle_mode):
        raise NotImplementedError()

    # endregion

    def iter_by_rows(self, contents_provider, doc_ids_iter, idle_mode):
        assert(isinstance(contents_provider, ContentsProvider))
        assert(isinstance(doc_ids_iter, collections.Iterable))

        for linked_data in contents_provider.from_doc_ids(doc_ids=doc_ids_iter, idle_mode=idle_mode):
            assert(isinstance(linked_data, LinkedDataWrapper))
            assert(isinstance(linked_data.Tag, ParsedNewsService))

            parsed_news_service = linked_data.Tag

            rows_it = self._provide_rows(parsed_news=parsed_news_service.ParsedNews,
                                         entity_service=parsed_news_service.get_provider(EntityServiceProvider.NAME),
                                         text_opinion_linkage=linked_data,
                                         idle_mode=idle_mode)

            for row in rows_it:
                yield linked_data.RelatedDocID, row
