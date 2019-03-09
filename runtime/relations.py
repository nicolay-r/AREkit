from core.source.entity import EntityCollection
from core.source.news import News
from core.source.opinion import Opinion


class RelationCollection:

    def __init__(self, relation_list):
        assert(isinstance(relation_list, list))
        self.relations = relation_list

    @classmethod
    def from_news_opinion(cls, news, opinion, debug=False):
        """
        lemmatize_to_str_func: function
            non lemmatized (unicode) -> (unicode), lemmatized string
        """
        assert(isinstance(news, News))
        assert(isinstance(opinion, Opinion))

        entities = news.entities
        assert(isinstance(entities, EntityCollection))

        left_entities = entities.try_get_entities(
            opinion.value_left, group_key=EntityCollection.KeyType.BY_SYNONYMS)
        right_entities = entities.try_get_entities(
            opinion.value_right, group_key=EntityCollection.KeyType.BY_SYNONYMS)

        if left_entities is None:
            if debug:
                print "Appropriate entity for '{}'->'...' has not been found".format(
                    opinion.value_left.encode('utf-8'))
            return cls(relation_list=[])

        if right_entities is None:
            if debug:
                print "Appropriate entity for '...'->'{}' has not been found".format(
                    opinion.value_right.encode('utf-8'))
            return cls(relation_list=[])

        relations = []
        for entity_left in left_entities:
            for entity_right in right_entities:
                relation = Relation(entity_left_ID=entity_left.ID,
                                    entity_right_ID=entity_right.ID,
                                    entity_by_id_func=lambda id: entities.get_entity_by_id(id))
                relations.append(relation)

        return cls(relations)

    def apply_filter(self, filter_function):
        self.relations = [r for r in self.relations if filter_function(r)]

    def __getitem__(self, item):
        assert(isinstance(item,  int))
        return self.relations[item]

    def __len__(self):
        return len(self.relations)

    def __iter__(self):
        for r in self.relations:
            yield r


class Relation:
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
        entity = self.__entity_by_id_func(self.__entity_left_ID)
        return entity.value

    @property
    def RigthEntityValue(self):
        entity = self.__entity_by_id_func(self.__entity_left_ID)
        return entity.value
