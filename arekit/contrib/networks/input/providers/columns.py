from arekit.common.data.input.providers.columns.sample import SampleColumnsProvider
from arekit.contrib.networks.input import const


class NetworkSampleColumnsProvider(SampleColumnsProvider):

    def get_columns_list_with_types(self):
        dtypes_list = super(NetworkSampleColumnsProvider, self).get_columns_list_with_types()

        # insert indices
        dtypes_list.append((const.FrameVariantIndices, str))
        dtypes_list.append((const.FrameConnotations, str))
        dtypes_list.append((const.SynonymSubject, str))
        dtypes_list.append((const.SynonymObject, str))
        dtypes_list.append((const.PosTags, str))

        return dtypes_list
