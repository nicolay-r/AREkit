import numpy as np

from core.relations import Relation
from base import Base


class PatternFeature(Base):

    def __init__(self, patterns, max_sentence_range=5):
        assert(type(patterns) == list)
        self.patterns = patterns
        self.max_sentence_range = max_sentence_range

    def create(self, relation):
        """ Get an amount of patterns between entities of relation
        """
        assert(isinstance(relation, Relation))
        e1 = relation.news.entities.get_entity_by_id(relation.entity_left_ID)
        e2 = relation.news.entities.get_entity_by_id(relation.entity_right_ID)

        text = relation.news.processed.get_text_between_entities_to_str(e1, e2)
        v = [text.count(p) for p in self.patterns]
        return np.array(v)
