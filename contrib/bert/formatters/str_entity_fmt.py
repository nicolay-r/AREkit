# -*- coding: utf-8 -*-
from arekit.common.entities.entity_mask import StringEntitiesFormatter
from arekit.common.entities.types import EntityType


class RussianEntitiesFormatter(StringEntitiesFormatter):

    def to_string(self, entity_type):
        assert(isinstance(entity_type, EntityType))

        if entity_type == EntityType.Object:
            return u"объект"
        if entity_type == EntityType.Subject:
            return u"субъект"
        if entity_type == EntityType.Other:
            return u"сущность"


