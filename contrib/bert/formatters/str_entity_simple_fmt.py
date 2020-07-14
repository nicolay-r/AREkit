from arekit.common.entities.entity_mask import StringEntitiesFormatter
from arekit.common.entities.types import EntityType


class EntitiesSimpleFormatter(StringEntitiesFormatter):

    def to_string(self, original_value, entity_type):
        assert(isinstance(entity_type, EntityType))

        if (entity_type == EntityType.Object) or (entity_type == EntityType.SynonymObject):
            return u"O"
        elif (entity_type == EntityType.Subject) or (entity_type == EntityType.SynonymSubject):
            return u"S"
        elif entity_type == EntityType.Other:
            return u"E"
