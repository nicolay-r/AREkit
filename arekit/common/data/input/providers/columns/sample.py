from arekit.common.data import const
from arekit.common.data.input.providers.columns.base import BaseColumnsProvider


class SampleColumnsProvider(BaseColumnsProvider):
    """
    [id, label, text_a] -- for train
    [id, text_a] -- for test
    """

    def __init__(self, store_labels):
        super(SampleColumnsProvider, self).__init__()
        self.__store_labels = store_labels
        self.__text_column_names = None

    # region properties

    @property
    def StoreLabels(self):
        return self.__store_labels

    @property
    def TextColumnNames(self):
        return self.__text_column_names

    # endregion

    def get_columns_list_with_types(self):
        """
        Composing df with the following columns:
            [id, label, type, text_a]
        """
        dtypes_list = super(SampleColumnsProvider, self).get_columns_list_with_types()

        dtypes_list.append((const.ID, str))
        dtypes_list.append((const.DOC_ID, str))

        # insert labels
        if self.__store_labels:
            dtypes_list.append((const.LABEL_UINT, 'int32'))
            dtypes_list.append((const.LABEL_STR, str))

        # insert text columns
        for col_name in self.__text_column_names:
            dtypes_list.append((col_name, str))

        # insert indices
        dtypes_list.append((const.S_IND, 'int32'))
        dtypes_list.append((const.T_IND, 'int32'))

        # opinion-extraction task related fields
        dtypes_list.append((const.OPINION_ID, 'int32'))
        dtypes_list.append((const.OPINION_LINKAGE_ID, 'int32'))

        return dtypes_list

    def set_text_column_names(self, text_column_names):
        assert(isinstance(text_column_names, list))
        self.__text_column_names = text_column_names
