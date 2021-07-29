from arekit.common.experiment import const
from arekit.common.experiment.input.readers.sample import InputSampleReader
from arekit.common.experiment.output.multiple import MulticlassOutput


class GoogleBertMulticlassOutput(MulticlassOutput):
    """ This output assumes to be provided with only labels
        by default proposed here:
        https://github.com/google-research/bert

        In addition to such output we provide the following parameters via samples_reader instance:
            - id -- is a row identifier, which is compatible with row_inds in serialized opinions.
            - news_id -- is a related news_id towards which the related output corresponds to.
    """

    def __init__(self, samples_reader, labels_scaler, has_output_header):
        assert(isinstance(samples_reader, InputSampleReader))
        super(GoogleBertMulticlassOutput, self).__init__(labels_scaler=labels_scaler,
                                                         has_output_header=has_output_header)
        self.__samples_reader = samples_reader

    # region protected methods

    def _csv_to_dataframe(self, filepath):
        df = super(GoogleBertMulticlassOutput, self)._csv_to_dataframe(filepath=filepath)

        # Exporting such information from samples.
        row_ids = self.__samples_reader.extract_ids()
        news_ids = self.__samples_reader.iter_news_ids()

        assert(len(row_ids) == len(news_ids) == len(df))

        # Providing the latter into output.
        df.insert(0, const.ID, row_ids)
        df.insert(1, const.NEWS_ID, news_ids)

        # Providing columns
        df.set_index(const.ID)

        df.columns = [str(c) for c in df.columns]

        return df

    # endregion
