from os.path import join, dirname
import unittest

from arekit.common.experiment.input.readers.samples import BaseInputSampleReader
from arekit.common.experiment.input.storages.base import BaseRowsStorage
from arekit.common.experiment.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.output.formatters.multiple import MulticlassOutputFormatter
from arekit.contrib.bert.output.google_bert_provider import GoogleBertOutputProvider
from tests.contrib.bert.labels import TestThreeLabelScaler


class TestOutputFormatters(unittest.TestCase):

    __current_dir = dirname(__file__)
    __input_samples_filepath = join(__current_dir, "data/test_sample_3l.tsv.gz")
    __google_bert_output_filepath_sample = join(__current_dir, "data/test_google_bert_output_3l.tsv")

    def test_google_bert_output_formatter(self):
        row_ids_provider = MultipleIDProvider()

        samples_reader = BaseInputSampleReader(
            storage=BaseRowsStorage.from_tsv(filepath=self.__input_samples_filepath),
            row_ids_provider=row_ids_provider)

        output = MulticlassOutputFormatter(
            labels_scaler=TestThreeLabelScaler(),
            output_provider=GoogleBertOutputProvider(samples_reader=samples_reader,
                                                     has_output_header=False))

        output.load(source=self.__google_bert_output_filepath_sample)

        df = output._df

        print(df)
        print(df.columns)

        # Check that all columns has string type.
        for c in df.columns:
            self.assertTrue(isinstance(c, str))


if __name__ == '__main__':
    unittest.main()
