from arekit.common.experiment import const
from arekit.common.experiment.input.readers.base_sample import BaseInputSampleReader
from arekit.contrib.networks.core.input.rows_parser import ParsedSampleRow


class NetworkInputSampleReaderHelper:

    @staticmethod
    def iter_uint_labeled_sample_rows(samples_reader):
        """
        Iter sample_ids with the related labels (if the latter presented in dataframe)
        """
        assert(isinstance(samples_reader, BaseInputSampleReader))

        for row_index, row in samples_reader.iter_rows():
            parsed_row = ParsedSampleRow(row)
            yield parsed_row.SampleID, parsed_row.UintLabel

    @staticmethod
    def calculate_news_id_by_sample_id_dict(samples_reader):
        """
        Creates sample_id -> news_id map.
        """
        assert(isinstance(samples_reader, BaseInputSampleReader))

        news_id_by_sample_id = {}

        for row_index, row in samples_reader.iter_rows():

            sample_id = row[const.ID]

            if sample_id in news_id_by_sample_id:
                continue

            news_id_by_sample_id[sample_id] = row[const.NEWS_ID]

        return news_id_by_sample_id
