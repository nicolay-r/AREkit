from arekit.common.experiment.input.providers.columns.sample import SampleColumnsProvider
from arekit.contrib.networks.core.input import const


class NetworkSampleColumnsProvider(SampleColumnsProvider):

    def get_columns_list_with_types(self):
        dtypes_list = super(NetworkSampleColumnsProvider, self).get_columns_list_with_types()

        # insert indices
        dtypes_list.append((const.FrameVariantIndices, str))
        dtypes_list.append((const.FrameRoles, str))
        dtypes_list.append((const.SynonymSubject, str))
        dtypes_list.append((const.SynonymObject, str))
        dtypes_list.append((const.Entities, str))
        dtypes_list.append((const.PosTags, str))

        return dtypes_list
