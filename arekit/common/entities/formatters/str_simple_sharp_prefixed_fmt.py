from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import EntityType


class SharpPrefixedEntitiesSimpleFormatter(StringEntitiesFormatter):

    def to_string(self, original_value, entity_type):
        assert(isinstance(entity_type, EntityType))

        if (entity_type == EntityType.Object) or (entity_type == EntityType.SynonymObject):
            return "#O"
        elif (entity_type == EntityType.Subject) or (entity_type == EntityType.SynonymSubject):
            return "#S"
        elif entity_type == EntityType.Other:
            return "#E"
