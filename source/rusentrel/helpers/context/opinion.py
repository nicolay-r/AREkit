class RuSentRelContextOpinion:
    """
    Strict Relation between two Entities
    """

    def __init__(self, entity_left_ID, entity_right_ID, entity_by_id_func):
        assert(isinstance(entity_left_ID, unicode))
        assert(isinstance(entity_right_ID, unicode))
        assert(callable(entity_by_id_func))
        self.__entity_left_ID = entity_left_ID
        self.__entity_right_ID = entity_right_ID
        self.__entity_by_id_func = entity_by_id_func

    @property
    def LeftEntityID(self):
        return self.__entity_left_ID

    @property
    def RightEntityID(self):
        return self.__entity_right_ID

    @property
    def LeftEntity(self):
        return self.__entity_by_id_func(self.__entity_left_ID)

    @property
    def RightEntity(self):
        return self.__entity_by_id_func(self.__entity_right_ID)

    @property
    def LeftEntityValue(self):
        return self.__entity_by_id_func(self.__entity_left_ID).value

    @property
    def RightEntityValue(self):
        return self.__entity_by_id_func(self.__entity_right_ID).value