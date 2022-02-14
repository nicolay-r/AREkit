class EntityCollection(object):
    """ Collection of annotated entities
    """

    class KeyType:
        BY_SYNONYMS = 0
        BY_VALUE = 1

    def __init__(self, entities, value_to_group_id_func):
        assert(isinstance(entities, list))
        assert(callable(value_to_group_id_func))

        self.__entities = entities
        self.__value_to_group_id_func = value_to_group_id_func

        self.__by_value = self.create_index(entities=entities,
                                            key_func=lambda e: e.Value)

        self.__by_synonyms = self.create_index(
            entities=entities,
            key_func=lambda e: value_to_group_id_func(e.Value))

    @staticmethod
    def __value_or_none(d, key):
        return d[key] if key in d else None

    # region protected methods

    def _sort_entities(self, key):
        assert(callable(key))
        self.__entities.sort(key=key)

    # endregion

    # region public methods

    @staticmethod
    def create_index(entities, key_func):
        index = {}
        for e in entities:
            key = key_func(e)
            if key in index:
                index[key].append(e)
            else:
                index[key] = [e]
        return index

    def get_entity_by_index(self, index):
        assert(isinstance(index, int))
        return self.__entities[index]

    def try_get_entities(self, value, group_key):
        assert(isinstance(value, str))

        if group_key == self.KeyType.BY_SYNONYMS:
            key = self.__value_to_group_id_func(value)
            return self.__value_or_none(self.__by_synonyms, key)
        if group_key == self.KeyType.BY_VALUE:
            return self.__value_or_none(self.__by_value, value)

    # endregion

    # region base methods

    def __len__(self):
        return len(self.__entities)

    def __iter__(self):
        for entity in self.__entities:
            yield entity

    # endregion
