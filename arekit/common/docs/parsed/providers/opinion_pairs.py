from arekit.common.docs.parsed.providers.base_pairs import BasePairProvider
from arekit.common.opinions.base import Opinion


class OpinionPairsProvider(BasePairProvider):

    NAME = "opinion-pairs-provider"

    def __init__(self, entity_value_func, **kwargs):
        super(OpinionPairsProvider, self).__init__(**kwargs)
        self.__entity_value_func = entity_value_func

    @property
    def Name(self):
        return self.NAME

    def _create_pair(self, source_entity, target_entity, label):
        return Opinion(source_value=self.__entity_value_func(source_entity),
                       target_value=self.__entity_value_func(target_entity),
                       label=label)
