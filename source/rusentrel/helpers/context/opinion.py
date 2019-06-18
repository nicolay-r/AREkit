from core.common.ref_opinon import RefOpinion
from core.evaluation.labels import NeutralLabel
from core.source.rusentrel.entities.collection import RuSentRelEntityCollection


class RuSentRelContextOpinion(RefOpinion):
    """
    Strict Relation between two Entities
    """

    def __init__(self,
                 e_source_doc_level_id,
                 e_target_doc_level_id,
                 doc_entities):
        assert(isinstance(e_source_doc_level_id, int))
        assert(isinstance(e_target_doc_level_id, int))
        assert(isinstance(doc_entities, RuSentRelEntityCollection))
        super(RuSentRelContextOpinion, self).__init__(source_id=e_source_doc_level_id,
                                                      target_id=e_target_doc_level_id,
                                                      sentiment=NeutralLabel(),
                                                      owner=doc_entities)

        self.__entity_left_ID = e_source_doc_level_id
        self.__entity_right_ID = e_target_doc_level_id
        self.__entity_by_id_func = doc_entities.get_entity_by_id

    @property
    def SourceEntity(self):
        return self.__entity_by_id_func(self.__entity_left_ID)

    @property
    def TargetEntity(self):
        return self.__entity_by_id_func(self.__entity_right_ID)

    @property
    def SourceEntityValue(self):
        return self.__entity_by_id_func(self.__entity_left_ID).Value

    @property
    def TargetEntityValue(self):
        return self.__entity_by_id_func(self.__entity_right_ID).Value