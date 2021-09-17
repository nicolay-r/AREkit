from arekit.common.experiment.input.storages.tsv_sample import TsvSampleStorage
from arekit.contrib.networks.core.input import const


# TODO. This should be refactored. #194.
# TODO. This should be refactored. #194.
# TODO. This should be refactored. #194.
class TsvNetworkSampleStorage(TsvSampleStorage):

    def _get_columns_list_with_types(self):
        dtypes_list = super(TsvNetworkSampleStorage, self)._get_columns_list_with_types()

        # insert indices
        dtypes_list.append((const.FrameVariantIndices, str))
        dtypes_list.append((const.FrameRoles, str))
        dtypes_list.append((const.SynonymSubject, str))
        dtypes_list.append((const.SynonymObject, str))
        dtypes_list.append((const.Entities, str))
        dtypes_list.append((const.PosTags, str))

        return dtypes_list
