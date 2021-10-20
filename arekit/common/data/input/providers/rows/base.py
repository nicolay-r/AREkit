import collections
import logging

from arekit.common.data.input.providers.opinions import OpinionProvider

logger = logging.getLogger(__name__)


class BaseRowProvider(object):
    """ Base provider for rows that suppose to be filled into BaseRowsStorage.
    """

    # region protected methods

    def _provide_rows(self, parsed_news, linked_wrapper, idle_mode):
        raise NotImplementedError()

    # endregion

    def iter_by_rows(self, opinion_provider, doc_ids_iter, idle_mode):
        assert(isinstance(opinion_provider, OpinionProvider))
        assert(isinstance(doc_ids_iter, collections.Iterable))

        for parsed_news, linked_wrapper in opinion_provider.iter_linked_opinion_wrappers(doc_ids_iter):

            rows_it = self._provide_rows(parsed_news=parsed_news,
                                         linked_wrapper=linked_wrapper,
                                         idle_mode=idle_mode)

            for row in rows_it:
                yield row