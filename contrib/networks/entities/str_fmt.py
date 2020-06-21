from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import EntityType


class StringSimpleMaskedEntityFormatter(StringEntitiesFormatter):
    """
    This entities formatter assumes to provide mask, which is depends on type of entity
    in order to prevent models from capturing information and making decisions onto
    frequencies of the related mentions in texts.
    """

    ENTITY_TYPE_SEPARATOR = u'_'

    def to_string(self, original_value, entity_type):
        assert(isinstance(original_value, Entity) or original_value is None)
        assert(isinstance(entity_type, EntityType))

        if entity_type == EntityType.Other:
            mask = u"ENTITY"
        elif entity_type == EntityType.Subject or entity_type == EntityType.SynonymSubject:
            mask = u"E_SUBJ"
        elif entity_type == EntityType.Object or entity_type == EntityType.SynonymObject:
            mask = u"E_OBJ"
        else:
            raise NotImplementedError()

        return mask
