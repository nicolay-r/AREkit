from arekit.common.experiment import const
from arekit.common.experiment.input.readers.sample import InputSampleReader


class NetworkInputSampleReader(InputSampleReader):

    def iter_labeled_sample_rows(self, label_scaler):
        """
        Iter sample_ids with the related labels (if the latter presented in dataframe)
        TODO. This might be also improved via rows_parser usage.
        """
        has_label = const.LABEL in self._df.columns

        for row_index, sample_id in self.__iter_rows():

            uint_label = int(self._df[const.LABEL][row_index]) if has_label else None

            if has_label:
                yield sample_id, label_scaler.uint_to_label(uint_label)
            else:
                yield sample_id, None

    def calculate_news_id_by_sample_id_dict(self):
        d = {}
        for row_index, sample_id in self.__iter_rows():

            if sample_id in d:
                continue

            d[sample_id] = self._df[const.NEWS_ID][row_index]

        return d

    def __iter_rows(self):
        for row_index, sample_id in enumerate(self._df[const.ID]):
            yield row_index, sample_id

    def iter_parsed_rows(self, label_scaler):
        # TODO. Iter parsed_rows (Instances from rows_parser.py).
        pass

