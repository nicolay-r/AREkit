from arekit.common.entities.types import EntityType


class StringEntitiesFormatter(object):

    def to_string(self, entity_type):
        assert(isinstance(entity_type, EntityType))
        raise NotImplementedError()