import collections
import logging

from arekit.common.data.input.providers.opinions import InputTextOpinionProvider
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
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

    # TODO. Limitation, this is now focused on linked text opinions only!
    # TODO. In general, this might be any provider, not only opinion-related.
    def iter_by_rows(self, opinion_provider, doc_ids_iter, idle_mode):
        assert(isinstance(opinion_provider, InputTextOpinionProvider))
        assert(isinstance(doc_ids_iter, collections.Iterable))

        for linkage in opinion_provider.iter_linked_opinions(doc_ids=doc_ids_iter, idle_mode=idle_mode):
            assert(isinstance(linkage, TextOpinionsLinkage))
            assert(isinstance(linkage.Tag, ParsedNewsService))

            parsed_news_service = linkage.Tag

            rows_it = self._provide_rows(parsed_news=parsed_news_service.ParsedNews,
                                         entity_service=parsed_news_service.get_provider(EntityServiceProvider.NAME),
                                         text_opinion_linkage=linkage,
                                         idle_mode=idle_mode)

            for row in rows_it:
                yield row
