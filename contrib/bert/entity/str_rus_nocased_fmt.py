# -*- coding: utf-8 -*-
from arekit.common.entities.str_mask_fmt import StringEntitiesFormatter
from arekit.common.entities.types import EntityType


class RussianEntitiesFormatter(StringEntitiesFormatter):

    def to_string(self, original_value, entity_type):
        assert(isinstance(entity_type, EntityType))

        if (entity_type == EntityType.Object) or (entity_type == EntityType.SynonymObject):
            return u"объект"
        elif (entity_type == EntityType.Subject) or (entity_type == EntityType.SynonymSubject):
            return u"субъект"
        if entity_type == EntityType.Other:
            return u"сущность"
