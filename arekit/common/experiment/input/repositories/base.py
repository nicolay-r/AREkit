import collections

from arekit.common.experiment.input.providers.columns.base import BaseColumnsProvider
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.input.providers.rows.base import BaseRowProvider
from arekit.common.experiment.input.storages.base import BaseRowsStorage
from arekit.common.experiment.input.writers.base import BaseWriter


class BaseInputRepository(object):

    def __init__(self, columns_provider, rows_provider, storage, writer=None):
        assert(isinstance(columns_provider, BaseColumnsProvider))
        assert(isinstance(rows_provider, BaseRowProvider))
        assert(isinstance(storage, BaseRowsStorage))
        assert(isinstance(writer, BaseWriter) or writer is None)

        self._columns_provider = columns_provider
        self._rows_provider = rows_provider
        self._storage = storage
        self._writer = writer

        # Do setup operations.
        self._setup_columns_provider()
        self._setup_rows_provider()

    # region protected methods

    def _setup_columns_provider(self):
        pass

    def _setup_rows_provider(self):
        pass

    # endregion

    def populate(self, opinion_provider, doc_ids_iter, target, desc=""):
        assert(isinstance(opinion_provider, OpinionProvider))
        assert(isinstance(self._storage, BaseRowsStorage))
        assert(isinstance(doc_ids_iter, collections.Iterable))

        def iter_rows(idle_mode):
            return self._rows_provider.iter_by_rows(
                opinion_provider=opinion_provider,
                doc_ids_iter=doc_ids_iter,
                idle_mode=idle_mode)

        self._storage.init_empty(columns_provider=self._columns_provider)

        self._storage.fill(lambda idle_mode: iter_rows(idle_mode),
                           columns_provider=self._columns_provider,
                           desc=desc)

        if self._writer is None:
            return

        # Write the contents of the storage into target.
        self._writer.save(self._storage, target)

        # After writing we free the contents of the storage.
        self._storage.free()
