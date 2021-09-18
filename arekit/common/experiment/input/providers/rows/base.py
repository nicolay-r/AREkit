import logging
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.input.storages.base import BaseRowsStorage
from arekit.common.utils import progress_bar_defined, progress_bar_iter

logger = logging.getLogger(__name__)


class BaseRowProvider(object):
    """ Base provider for rows that suppose to be filled into BaseRowsStorage.
    """

    def __init__(self, storage):
        assert(isinstance(storage, BaseRowsStorage))
        self._storage = storage

    def __iter_by_rows(self, opinion_provider, idle_mode):
        assert(isinstance(opinion_provider, OpinionProvider))

        for parsed_news, linked_wrapper in opinion_provider.iter_linked_opinion_wrappers():

            rows_it = self._provide_rows(parsed_news=parsed_news,
                                         linked_wrapper=linked_wrapper,
                                         idle_mode=idle_mode)

            for row in rows_it:
                yield row

    def _provide_rows(self, parsed_news, linked_wrapper, idle_mode):
        raise NotImplementedError()

    def format(self, opinion_provider, desc=""):
        assert(isinstance(opinion_provider, OpinionProvider))

        logged_rows_it = progress_bar_iter(self.__iter_by_rows(opinion_provider, idle_mode=True),
                                           desc="Calculating rows count",
                                           unit="rows")
        rows_count = sum(1 for _ in logged_rows_it)

        logger.info("Filling with blank rows: {}".format(rows_count))
        self._storage.fill_with_blank_rows(rows_count)
        logger.info("Completed!")

        it = progress_bar_defined(iterable=self.__iter_by_rows(opinion_provider, idle_mode=False),
                                  desc="{fmt}".format(fmt=desc),
                                  total=rows_count)

        for row_index, row in enumerate(it):
            for column, value in row.items():
                self._storage.set_value(row_ind=row_index,
                                        column=column,
                                        value=value)

        self._storage.log_info()

    def save(self):
        self._storage.save()
