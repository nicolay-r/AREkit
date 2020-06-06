# -*- coding: utf-8 -*-
from arekit.common.entities.entity_mask import StringEntitiesFormatter
from arekit.common.entities.types import EntityType


class RussianEntitiesFormatter(StringEntitiesFormatter):

    def to_string(self, entity_type):
        assert(isinstance(entity_type, EntityType))

        if (entity_type == EntityType.Object) or (entity_type == EntityType.SynonymObject):
            return u"объект"
        elif (entity_type == EntityType.Subject) or (entity_type == EntityType.SynonymSubject):
            return u"субъект"
        elif entity_type == EntityType.Other:
            return u"сущность"



