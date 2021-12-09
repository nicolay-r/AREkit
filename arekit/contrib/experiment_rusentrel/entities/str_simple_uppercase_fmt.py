from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import EntityType


class SimpleUppercasedEntityFormatter(StringEntitiesFormatter):

    def to_string(self, original_value, entity_type):
        assert(isinstance(original_value, Entity) or original_value is None)
        assert(isinstance(entity_type, EntityType))

        if entity_type == EntityType.Other:
            mask = "ENTITY"
        elif entity_type == EntityType.Subject or entity_type == EntityType.SynonymSubject:
            mask = "E_SUBJ"
        elif entity_type == EntityType.Object or entity_type == EntityType.SynonymObject:
            mask = "E_OBJ"
        else:
            raise NotImplementedError()

        return mask
