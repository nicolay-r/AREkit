
class BaseColumnsProvider(object):

    ROW_ID = 'row_id'

    def get_columns_list_with_types(self):
        dtypes_list = list()
        dtypes_list.append((BaseColumnsProvider.ROW_ID, 'int32'))
        return dtypes_list
