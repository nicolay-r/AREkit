from arekit.common.labels.base import Label
from arekit.common.linked.data import LinkedDataWrapper
from arekit.common.opinions.base import Opinion


class LinkedOpinionWrapper(LinkedDataWrapper):

    @staticmethod
    def _aggregate_by_first(item, label):
        assert(isinstance(item, Opinion))
        assert(isinstance(label, Label))
        return Opinion(source_value=item.SourceValue,
                       target_value=item.TargetValue,
                       sentiment=label)
