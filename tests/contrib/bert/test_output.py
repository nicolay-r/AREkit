from os.path import join, dirname
import unittest

from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.contrib.bert.output.google_bert_provider import GoogleBertOutputStorage


class TestOutputFormatters(unittest.TestCase):

    __current_dir = dirname(__file__)
    __input_samples_filepath = join(__current_dir, "data/test_sample_3l.tsv.gz")
    __google_bert_output_filepath_sample = join(__current_dir, "data/test_google_bert_output_3l.tsv")

    def test_google_bert_output_formatter(self):
        row_ids_provider = MultipleIDProvider()

        samples_view = BaseSampleStorageView(
            storage=BaseRowsStorage.from_tsv(filepath=self.__input_samples_filepath),
            row_ids_provider=row_ids_provider)

        # Initialize storage.
        output_storage = GoogleBertOutputStorage.from_tsv(
            filepath=self.__google_bert_output_filepath_sample,
            header=None)
        output_storage.apply_samples_view(samples_view)

        df = output_storage.DataFrame

        print(df)
        print(df.columns)

        # Check that all columns has string type.
        for c in df.columns:
            self.assertTrue(isinstance(c, str))


if __name__ == '__main__':
    unittest.main()
