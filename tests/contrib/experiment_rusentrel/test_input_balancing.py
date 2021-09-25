import sys
import unittest


sys.path.append('../')

from arekit.common.experiment import const
from arekit.contrib.bert.input.providers.label_binary import BinaryLabelProvider
from arekit.contrib.experiment_rusentrel.common import entity_to_group_func
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.common.experiment.input.providers.columns.sample import SampleColumnsProvider
from arekit.common.experiment.input.repositories.sample import BaseInputSamplesRepository
from arekit.common.experiment.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.experiment.input.storages.sample import BaseSampleStorage
from arekit.common.experiment.input.providers.text.single import BaseSingleTextProvider
from arekit.common.experiment.input.terms_mapper import OpinionContainingTextTermsMapper
from arekit.common.entities.formatters.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.processing.lemmatization.mystem import MystemWrapper


class TestInputBalancing(unittest.TestCase):

    def test_balancing(self):

        stemmer = MystemWrapper()
        label_provider = BinaryLabelProvider(label_scaler=ThreeLabelScaler())
        synonyms = RuSentRelSynonymsCollectionProvider.load_collection(stemmer=stemmer,
                                                                       is_read_only=True)
        terms_mapper = OpinionContainingTextTermsMapper(
            entity_formatter=StringEntitiesSimpleFormatter(),
            entity_to_group_func=lambda entity: entity_to_group_func(entity=entity,
                                                                     synonyms=synonyms))
        text_provider = BaseSingleTextProvider(terms_mapper)

        storage = BaseSampleStorage()
        BaseSampleRowProvider(label_provider=label_provider,
                              text_provider=text_provider)

        BaseInputSamplesRepository(
            columns_provider=SampleColumnsProvider(store_labels=True),
            rows_provider=BaseSampleRowProvider(
                label_provider=label_provider,
                text_provider=text_provider),
            storage=storage
        )

        storage.init_empty()

        df = storage._df

        df = df.append({"row_id": 1, "id": 1, "label": 0, "text_a": "-", "s_ind": 0, "t_ind": 0}, ignore_index=True)
        df = df.append({"row_id": 1, "id": 2, "label": 1, "text_a": "-", "s_ind": 0, "t_ind": 0}, ignore_index=True)
        df = df.append({"row_id": 1, "id": 5, "label": 0, "text_a": "-", "s_ind": 0, "t_ind": 0}, ignore_index=True)
        df = df.append({"row_id": 1, "id": 6, "label": 0, "text_a": "-", "s_ind": 0, "t_ind": 0}, ignore_index=True)

        storage._df = df

        print("Original:")
        print(storage._df.shape)

        storage._balance(column_name=const.LABEL)

        print("Balanced:")
        print(storage._df)


if __name__ == '__main__':
    unittest.main()
