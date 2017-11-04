import numpy as np
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
        e1 = relation.news.entities.get_by_ID(relation.entity_left_ID)
        e2 = relation.news.entities.get_by_ID(relation.entity_right_ID)

        s1 = relation.news.get_sentence_by_entity(e1).index
        s2 = relation.news.get_sentence_by_entity(e2).index

        preps = self._get_prepositions_count(s1, s2, e1, e2, relation.news)

        return np.array([preps])

    def _get_prepositions_count(self, s1, s2, e1, e2, news):
        r = 0

        range_from = min(e1.end, e2.end)
        range_to = max(e1.begin, e2.begin)

        for s_id in range(s1, s2+1):
            s = news.sentences[s_id]
            for p in self.prepositions:
                has_prep = self._has_sentence_preposition(
                    s.text.lower(), s.begin, s.end, p, range_from, range_to)
                r += 1 if has_prep else 0

        return r

    @staticmethod
    def _has_sentence_preposition(s_text, s_begin, s_end, prep, range_from, range_to):
        assert(type(s_text) == unicode)
        assert(type(prep) == unicode)

        index = s_text.find(prep)

        if (index == -1):
            return False

        p_from = s_begin + index
        p_to = p_from + len(prep)

        if (p_from > range_from and p_to < range_to):
            return True

        return False
