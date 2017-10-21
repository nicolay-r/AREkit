from feature import Feature
from core.source.relations import Relation


class PrepositionsCountFeature(Feature):

    def __init__(self, prepositions):
        assert(type(prepositions) == list)
        self.prepositions = prepositions

    def create(self, relation):
        """ Get an amount of prepositions between relation entities
        """
        assert(isinstance(relation, Relation))
        e1 = relation.news.entities.find_by_ID(relation.entity_left_ID)
        e2 = relation.news.entities.find_by_ID(relation.entity_right_ID)

        s1 = relation.news.find_sentence_by_entity(e1)
        s2 = relation.news.find_sentence_by_entity(e2)

        # TODO

        return []
