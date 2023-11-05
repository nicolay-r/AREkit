from collections import Counter
from collections.abc import Iterable
import logging

from arekit.common.data.input.providers.contents import ContentsProvider
from arekit.common.linkage.base import LinkedDataWrapper
from arekit.common.docs.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.linkage.meta import MetaEmptyLinkedDataWrapper

logger = logging.getLogger(__name__)


class BaseRowProvider(object):
    """ Base provider for rows that suppose to be filled into BaseRowsStorage.
    """

    def __init__(self):
        self.__rows_counter = None

    # region protected methods

    # TODO. This might be also generalized.
    # TODO. Idle-mode is also a implementation and task specific parameter, i.e. might be removed from here.
    def _provide_rows(self, parsed_doc, entity_service, text_opinion_linkage, idle_mode):
        raise NotImplementedError()

    def _count_row(self):
        index = self.__rows_counter["rows_iterated"]
        self.__rows_counter["rows_iterated"] += 1
        return index

    # endregion

    def __iter_rows(self, linked_data, idle_mode):
        parsed_doc_service = linked_data.Tag
        return self._provide_rows(parsed_doc=parsed_doc_service.ParsedDocument,
                                  entity_service=parsed_doc_service.get_provider(EntityServiceProvider.NAME),
                                  text_opinion_linkage=linked_data,
                                  idle_mode=idle_mode)

    def iter_by_rows(self, contents_provider, doc_ids_iter, idle_mode):
        assert(isinstance(contents_provider, ContentsProvider))
        assert(isinstance(doc_ids_iter, Iterable))

        self.__rows_counter = Counter()

        for linked_data in contents_provider.from_doc_ids(doc_ids=doc_ids_iter, idle_mode=idle_mode):
            assert(isinstance(linked_data, LinkedDataWrapper))

            if isinstance(linked_data, MetaEmptyLinkedDataWrapper):
                if idle_mode:
                    # In the case of the IDLE mode we do not consider the meta-data.
                    data_it = []
                else:
                    # Consider the actual linked data instance.
                    data_it = [linked_data]
            else:
                # Consider the actual rows of the related linked data.
                data_it = self.__iter_rows(linked_data=linked_data, idle_mode=idle_mode)

            for data in data_it:
                yield linked_data.RelatedDocID, data

        self.__rows_counter = None
