from arekit.common.experiment import const
from arekit.common.experiment.output.tsv_provider import TsvBaseOutputProvider


class GoogleBertOutputProvider(TsvBaseOutputProvider):
    """ This output assumes to be provided with only labels
        by default proposed here:
        https://github.com/google-research/bert

        In addition to such output we provide the following parameters via samples_reader instance:
            - id -- is a row identifier, which is compatible with row_inds in serialized opinions.
            - news_id -- is a related news_id towards which the related output corresponds to.
    """

    def __init__(self, samples_reader, has_output_header):
        super(GoogleBertOutputProvider, self).__init__(has_output_header=has_output_header)
        self.__samples_reader = samples_reader

    # region protected methods

    def _csv_to_dataframe(self, filepath):
        df = super(GoogleBertOutputProvider, self)._csv_to_dataframe(filepath=filepath)

        # Exporting such information from samples.
        row_ids = self.__samples_reader.extract_ids()
        news_ids = self.__samples_reader.extract_news_ids()

        assert(len(row_ids) == len(news_ids) == len(df))

        # Providing the latter into output.
        df.insert(0, const.ID, row_ids)
        df.insert(1, const.NEWS_ID, news_ids)

        # Providing columns
        df.set_index(const.ID)

        df.columns = [str(c) for c in df.columns]

        return df

    # endregion
