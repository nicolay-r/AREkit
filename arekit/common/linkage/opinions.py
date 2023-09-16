from arekit.common.linkage.base import LinkedDataWrapper
from arekit.common.opinions.base import Opinion


class OpinionsLinkage(LinkedDataWrapper):

    def _get_data_label(self, item):
        assert(isinstance(item, Opinion))
        return item.Label
