from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import OpinionEntityType


class SharpPrefixedEntitiesSimpleFormatter(StringEntitiesFormatter):

    def to_string(self, original_value, entity_type):
        assert(isinstance(entity_type, OpinionEntityType))

        if (entity_type == OpinionEntityType.Object) or (entity_type == OpinionEntityType.SynonymObject):
            return "#O"
        elif (entity_type == OpinionEntityType.Subject) or (entity_type == OpinionEntityType.SynonymSubject):
            return "#S"
        elif entity_type == OpinionEntityType.Other:
            return "#E"
