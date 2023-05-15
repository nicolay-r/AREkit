from arekit.common.data.input.providers.columns.base import BaseColumnsProvider
from arekit.common.data.input.providers.contents import ContentsProvider
from arekit.common.data.input.providers.rows.base import BaseRowProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.data.writers.base import BaseWriter


class BaseInputRepository(object):

    def __init__(self, columns_provider, rows_provider, storage):
        assert(isinstance(columns_provider, BaseColumnsProvider))
        assert(isinstance(rows_provider, BaseRowProvider))
        assert(isinstance(storage, BaseRowsStorage))

        self._columns_provider = columns_provider
        self._rows_provider = rows_provider
        self._storage = storage

        # Do setup operations.
        self._setup_columns_provider()
        self._setup_rows_provider()

    # region protected methods

    def _setup_columns_provider(self):
        pass

    def _setup_rows_provider(self):
        pass

    # endregion

    def populate(self, contents_provider, doc_ids, desc="", writer=None, target=None):
        assert(isinstance(contents_provider, ContentsProvider))
        assert(isinstance(self._storage, BaseRowsStorage))
        assert(isinstance(doc_ids, list))
        assert(isinstance(writer, BaseWriter) or writer is None)
        assert(isinstance(target, str) or target is None)

        def iter_rows(idle_mode):
            return self._rows_provider.iter_by_rows(
                contents_provider=contents_provider,
                doc_ids_iter=doc_ids,
                idle_mode=idle_mode)

        self._storage.init_empty(columns_provider=self._columns_provider)

        is_async_write_mode_on = writer is not None and target is not None

        if is_async_write_mode_on:
            writer.open_target(target)

        self._storage.fill(lambda idle_mode: iter_rows(idle_mode),
                           columns_provider=self._columns_provider,
                           row_handler=lambda: writer.commit_line(self._storage) if is_async_write_mode_on else None,
                           desc=desc)

        if is_async_write_mode_on:
            writer.close_target()

    def push(self, writer, target, free_storage=True):
        if not isinstance(self._storage, RowCacheStorage):
            writer.write_all(self._storage, target)

        # After writing we free the contents of the storage.
        if free_storage:
            self._storage.free()
