from arekit.common.entities.base import Entity
from arekit.common.news.parsed.providers.base_pairs import BasePairProvider
from arekit.common.opinions.base import Opinion


class OpinionPairsProvider(BasePairProvider):

    NAME = "opinion-pairs-provider"

    @property
    def Name(self):
        return self.NAME

    def _create_pair(self, source_entity, target_entity, label):
        assert(isinstance(source_entity, Entity))
        assert(isinstance(target_entity, Entity))

        return Opinion(source_value=source_entity.Value,
                       target_value=target_entity.Value,
                       sentiment=label)
