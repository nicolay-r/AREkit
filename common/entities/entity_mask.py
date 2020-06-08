from arekit.common.entities.types import EntityType


class StringEntitiesFormatter(object):

    def to_string(self, original_value, entity_type):
        assert(isinstance(entity_type, EntityType))
        raise NotImplementedError()


