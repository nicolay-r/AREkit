from arekit.common.linked.data import LinkedDataWrapper
from arekit.common.opinions.base import Opinion


class LinkedOpinionWrapper(LinkedDataWrapper):

    def _get_data_label(self, item):
        assert(isinstance(item, Opinion))
        return item.Sentiment
