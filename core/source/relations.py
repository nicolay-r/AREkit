from news import News


class Relation:
    """ Relations between two entities
    """

    def __init__(self, entity_left_ID, entity_right_ID, news):
        assert(type(entity_left_ID) == unicode)
        assert(type(entity_right_ID) == unicode)
        assert(isinstance(news, News))
        self.entity_left_ID = entity_left_ID
        self.entity_right_ID = entity_right_ID
        self.news = news
