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
        self._setup_storage()
        self._setup_writer()

    # region protected methods

    def _setup_columns_provider(self):
        pass

    def _setup_rows_provider(self):
        self._rows_provider.set_storage(self._storage)

    def _setup_storage(self):
        self._storage.set_columns_provider(self._columns_provider)

    def _setup_writer(self):
        if isinstance(self._writer, BaseWriter):
            self._writer.set_storage(self._storage)

    # endregion

    def populate(self, opinion_provider, target, desc=""):
        assert(isinstance(opinion_provider, OpinionProvider))
        assert(isinstance(self._storage, BaseRowsStorage))

        with self._storage:
            self._rows_provider.fill(opinion_provider, desc)

            if self._writer is None:
                return

            assert(isinstance(self._writer, BaseWriter))

            return self._writer.save(target)
