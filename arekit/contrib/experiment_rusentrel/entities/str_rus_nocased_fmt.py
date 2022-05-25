from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import OpinionEntityType


class RussianEntitiesFormatter(StringEntitiesFormatter):

    def to_string(self, original_value, entity_type):
        assert(isinstance(entity_type, OpinionEntityType))

        if (entity_type == OpinionEntityType.Object) or (entity_type == OpinionEntityType.SynonymObject):
            return "объект"
        elif (entity_type == OpinionEntityType.Subject) or (entity_type == OpinionEntityType.SynonymSubject):
            return "субъект"
        if entity_type == OpinionEntityType.Other:
            return "сущность"
