from arekit.common.experiment.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.experiment.input.repositories.base import BaseInputRepository


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

    def _setup_storage(self):
        """ Setup output labels uint
        """
        super(BaseInputSamplesRepository, self)._setup_storage()
        self._storage.set_output_labels_uint(
            labels_uint=self._rows_provider.LabelProvider.OutputLabelsUint)
