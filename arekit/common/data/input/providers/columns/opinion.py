from arekit.common.data import const
from arekit.common.data.input.providers.columns.base import BaseColumnsProvider


class OpinionColumnsProvider(BaseColumnsProvider):

    def get_columns_list_with_types(self):
        dtypes_list = super(OpinionColumnsProvider, self).get_columns_list_with_types()
        dtypes_list.append((const.ID, str))
        dtypes_list.append((const.DOC_ID, 'int32'))
        dtypes_list.append((const.SOURCE, str))
        dtypes_list.append((const.TARGET, str))
        return dtypes_list
