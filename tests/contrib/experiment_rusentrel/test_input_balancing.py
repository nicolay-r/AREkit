import sys
import unittest


sys.path.append('../')

from arekit.common.data.input.providers.columns.sample import SampleColumnsProvider
from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.data.input.repositories.sample import BaseInputSamplesRepository
from arekit.common.data.input.terms_mapper import OpinionContainingTextTermsMapper
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.contrib.bert.input.providers.label_binary import BinaryLabelProvider
from arekit.contrib.experiment_rusentrel.entities.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler


class TestInputBalancing(unittest.TestCase):

    def test_balancing(self):

        label_provider = BinaryLabelProvider(label_scaler=ThreeLabelScaler())
        terms_mapper = OpinionContainingTextTermsMapper(
            entity_formatter=StringEntitiesSimpleFormatter())
        text_provider = BaseSingleTextProvider(terms_mapper)

        storage = BaseRowsStorage()

        columns_provider = SampleColumnsProvider(store_labels=True)

        samples_repo = BaseInputSamplesRepository(
            columns_provider=columns_provider,
            rows_provider=BaseSampleRowProvider(
                label_provider=label_provider,
                text_provider=text_provider),
            storage=storage)

        storage.init_empty(columns_provider)

        df = storage._df

        df = df.append({"row_id": 1, "id": 1, "label": 0, "text_a": "-", "s_ind": 0, "t_ind": 0}, ignore_index=True)
        df = df.append({"row_id": 1, "id": 2, "label": 1, "text_a": "-", "s_ind": 0, "t_ind": 0}, ignore_index=True)
        df = df.append({"row_id": 1, "id": 5, "label": 0, "text_a": "-", "s_ind": 0, "t_ind": 0}, ignore_index=True)
        df = df.append({"row_id": 1, "id": 6, "label": 0, "text_a": "-", "s_ind": 0, "t_ind": 0}, ignore_index=True)

        storage._df = df

        print("Original:")
        print(storage._df.shape)

        samples_repo.balance()

        print("Balanced:")
        print(storage._df)


if __name__ == '__main__':
    unittest.main()
