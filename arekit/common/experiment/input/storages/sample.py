import collections
import logging
from arekit.common.experiment import const
from arekit.common.experiment.input.storages.base import BaseRowsStorage


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseSampleStorage(BaseRowsStorage):
    """
    Custom Processor with the following fields

    [id, label, text_a] -- for train
    [id, text_a] -- for test
    """

    def __init__(self, store_labels):
        assert(isinstance(store_labels, bool))
        super(BaseSampleStorage, self).__init__()

        self.__store_labels = store_labels
        self.__text_column_names = None
        self._output_labels_uint = None

    @property
    def StoreLabels(self):
        return self.__store_labels

    # region protected methods

    def _get_columns_list_with_types(self):
        """
        Composing df with the following columns:
            [id, label, type, text_a]
        """
        dtypes_list = super(BaseSampleStorage, self)._get_columns_list_with_types()

        dtypes_list.append((const.ID, str))
        dtypes_list.append((const.NEWS_ID, 'int32'))

        # insert labels
        if self.__store_labels:
            dtypes_list.append((const.LABEL, 'int32'))

        # insert text columns
        for col_name in self.__text_column_names:
            dtypes_list.append((col_name, str))

        # insert indices
        dtypes_list.append((const.S_IND, 'int32'))
        dtypes_list.append((const.T_IND, 'int32'))

        return dtypes_list

    def _create_blank_df(self, size):
        df = self._create_empty_df()
        self._fast_init_df(df=df, rows_count=size)
        return df

    def _fast_init_df(self, df, rows_count):
        df[self.ROW_ID] = list(range(rows_count))
        df.set_index(self.ROW_ID, inplace=True)

    # endregion

    def set_output_labels_uint(self, labels_uint):
        if self._output_labels_uint is not None:
            raise Exception("Output labels already defined!")
        self._output_labels_uint = labels_uint

    def set_text_column_names(self, text_column_names):
        assert(isinstance(text_column_names, collections.Iterable))
        if self.__text_column_names is not None:
            raise Exception("Text column names already defined!")
        self.__text_column_names = list(text_column_names)

    def save(self):
        """ This might be implemented in nested classes.
            The default, i.e. pandas-based storage is not considered
            to be saved into the particular target.
        """
        pass

    def __len__(self):
        return len(self._df.index)
