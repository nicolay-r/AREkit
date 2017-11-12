from core.source.news import News
from core.source.synonyms import SynonymsCollection


class Relation:
    """ Strict Relation between two Entities
    """

    def __init__(self, entity_left_ID, entity_right_ID, news, synonyms):
        assert(type(entity_left_ID) == unicode)
        assert(type(entity_right_ID) == unicode)
        assert(isinstance(news, News))
        self.entity_left_ID = entity_left_ID
        self.entity_right_ID = entity_right_ID
        self.news = news
