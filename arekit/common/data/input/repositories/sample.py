import logging

from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.data.input.repositories.base import BaseInputRepository

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseInputSamplesRepository(BaseInputRepository):

    def _setup_rows_provider(self):
        """ Setup store labels.
        """
        assert(isinstance(self._rows_provider, BaseSampleRowProvider))
        self._rows_provider.set_store_labels(self._columns_provider.StoreLabels)

    def _setup_columns_provider(self):
        """ Setup text column names.
        """
        text_column_names = list(self._rows_provider.TextProvider.iter_columns())
        self._columns_provider.set_text_column_names(text_column_names)
