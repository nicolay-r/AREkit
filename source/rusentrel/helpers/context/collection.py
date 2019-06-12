from core.common.opinions.opinion import Opinion
from core.source.rusentrel.entities.collection import RuSentRelEntityCollection
from core.source.rusentrel.news import RuSentRelNews
from core.source.rusentrel.helpers.context.opinion import RuSentRelContextOpinion


class RuSentRelContextOpinionCollection:

    def __init__(self, relation_list):
        assert(isinstance(relation_list, list))
        self.__relations = relation_list

    @classmethod
    def from_news_opinion(cls, news, opinion, debug=False):
        assert(isinstance(news, RuSentRelNews))
        assert(isinstance(opinion, Opinion))

        entities = news.Entities
        assert(isinstance(entities, RuSentRelEntityCollection))

        left_entities = entities.try_get_entities(
            opinion.value_left, group_key=RuSentRelEntityCollection.KeyType.BY_SYNONYMS)
        right_entities = entities.try_get_entities(
            opinion.value_right, group_key=RuSentRelEntityCollection.KeyType.BY_SYNONYMS)

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
                relation = RuSentRelContextOpinion(entity_left_ID=entity_left.ID,
                                                   entity_right_ID=entity_right.ID,
                                                   entity_by_id_func=entities.get_entity_by_id)
                relations.append(relation)

        return cls(relations)

    def apply_filter(self, filter_function):
        self.__relations = [r for r in self.__relations if filter_function(r)]

    def __getitem__(self, item):
        assert(isinstance(item,  int))
        return self.__relations[item]

    def __len__(self):
        return len(self.__relations)

    def __iter__(self):
        for relation in self.__relations:
            yield relation