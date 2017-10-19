from feature import Feature
from core.relations import Relation
from core.annot import Entity
from core.news import News


class PrepositionsCountFeature(Feature):

    def __init__(self, prepositions):
        assert(type(prepositions) == list)
        self.prepositions = prepositions

    # Duplicated
    def __find_sentence_with_entity(self, entity, news):
        assert(isinstance(entity, Entity))
        assert(isinstance(news, News))
        for i, sentence in enumerate(news.sentences):
            if sentence.has_entity(entity.ID):
                return i

        raise Exception("Can't find entity!")


    def create(self, relation):
        """ Get an amount of prepositions between relation entities
        """
        assert(isinstance(relation, Relation))
        e1 = relation.news.entities.find_by_ID(relation.entity_left_ID)
        e2 = relation.news.entities.find_by_ID(relation.entity_right_ID)

        s1 = self.__find_sentence_with_entity(e1, relation.news)
        s2 = self.__find_sentence_with_entity(e2, relation.news)

        # TODO

        return []
