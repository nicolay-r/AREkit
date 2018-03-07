from core.source.news import News
from core.source.opinion import Opinion


class RelationCollection:

    def __init__(self):
        pass

    def from_news_opinion(news, opinion, synonyms):

        def get_appropriate_entities_value(value):
            if synonyms.has_synonym(value):
                return filter(
                    lambda s: entities.has_entity_by_value(s),
                    synonyms.get_synonyms_list(value))

            elif entities.has_entity_by_value(value):
                return [value]
            else:
                return []

        assert(isinstance(news, News))
        assert(isinstance(opinion, Opinion))
        # TODO. Create method in core that provides all pairs of entities
        # related to a Opinion. Widely used.

    def __iter__(self):
        pass


class Relation:
    """ Strict Relation between two Entities
    """

    def __init__(self, entity_left_ID, entity_right_ID, news):
        assert(type(entity_left_ID) == unicode)
        assert(type(entity_right_ID) == unicode)
        assert(isinstance(news, News))
        self.entity_left_ID = entity_left_ID
        self.entity_right_ID = entity_right_ID
        self.news = news
