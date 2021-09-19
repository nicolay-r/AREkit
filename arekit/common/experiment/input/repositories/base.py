from arekit.common.experiment.input.providers.columns.base import BaseColumnsProvider
from arekit.common.experiment.input.providers.opinions import OpinionProvider
from arekit.common.experiment.input.providers.rows.base import BaseRowProvider
from arekit.common.experiment.input.storages.base import BaseRowsStorage


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
        self._setup_storage()

    # region protected methods

    def _setup_columns_provider(self):
        pass

    def _setup_rows_provider(self):
        self._rows_provider.set_storage(self._storage)

    def _setup_storage(self):
        self._storage.set_columns_provider(self._columns_provider)

    # endregion

    def populate(self, opinion_provider, target, desc=""):
        assert(isinstance(opinion_provider, OpinionProvider))
        assert(isinstance(self._storage, BaseRowsStorage))

        self._storage.init_empty()

        with self._storage as storage:
            self._rows_provider.format(opinion_provider, desc)
            return storage.save(target)
