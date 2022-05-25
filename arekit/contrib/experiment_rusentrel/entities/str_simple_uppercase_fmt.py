from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import OpinionEntityType


class SimpleUppercasedEntityFormatter(StringEntitiesFormatter):

    def to_string(self, original_value, entity_type):
        assert(isinstance(original_value, Entity) or original_value is None)
        assert(isinstance(entity_type, OpinionEntityType))

        if entity_type == OpinionEntityType.Other:
            mask = "ENTITY"
        elif entity_type == OpinionEntityType.Subject or entity_type == OpinionEntityType.SynonymSubject:
            mask = "E_SUBJ"
        elif entity_type == OpinionEntityType.Object or entity_type == OpinionEntityType.SynonymObject:
            mask = "E_OBJ"
        else:
            raise NotImplementedError()

        return mask
