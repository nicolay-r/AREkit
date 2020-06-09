# -*- coding: utf-8 -*-
from arekit.common.entities.base import Entity
from arekit.common.entities.entity_mask import StringEntitiesFormatter
from arekit.common.entities.types import EntityType
from arekit.processing.pos.russian import RussianPOSTagger


class RussianEntitiesFormatter(StringEntitiesFormatter):

    def __init__(self, pos_tagger):
        assert(isinstance(pos_tagger, RussianPOSTagger))
        self.__pos_tagger = pos_tagger

    def to_string(self, original_value, entity_type):
        assert(isinstance(original_value, Entity) or original_value is None)
        assert(isinstance(entity_type, EntityType))

        if (entity_type == EntityType.Object) or (entity_type == EntityType.SynonymObject):
            return u"объект"
        elif (entity_type == EntityType.Subject) or (entity_type == EntityType.SynonymSubject):
            return u"субъект"
        elif entity_type == EntityType.Other:
            return u"сущность"



