# -*- coding: utf-8 -*-
import sys
import unittest

from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.input.readers.sample import InputSampleReader
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.contrib.bert.output.google_bert import GoogleBertMulticlassOutput

sys.path.append('../../')


class TestOutputFormatters(unittest.TestCase):

    __input_samples_filepath = u"data/test_sample_3l.tsv.gz"

    __google_bert_output_filepath_sample = u"data/test_google_bert_output_3l.tsv"

    def test_google_bert_output_formatter(self):
        row_ids_provider = MultipleIDProvider()

        samples_reader = InputSampleReader.from_tsv(filepath=self.__input_samples_filepath,
                                                    row_ids_provider=row_ids_provider)

        output = GoogleBertMulticlassOutput(samples_reader=samples_reader,
                                            labels_scaler=ThreeLabelScaler(),
                                            has_output_header=False)

        output.init_from_tsv(self.__google_bert_output_filepath_sample)

        df = output._DataFrame

        print df.columns
        print df

        # Check that all columns has string type.
        for c in df.columns:
            self.assert_(isinstance(c, str))


if __name__ == '__main__':
    unittest.main()