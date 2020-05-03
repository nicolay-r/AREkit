
class News(object):

    def __init__(self, news_id):
        assert(isinstance(news_id, int))
        self.__news_id = news_id

    @property
    def ID(self):
        return self.__news_id

    def iter_wrapped_linked_text_opinions(self, opinions):
        """
        opinions: iterable Opinion
            is an iterable opinions that should be used to find a related text_opinion entries.
        """
        raise NotImplementedError()

    def parse(self, options):
        """
        returns: ParsedNews
        """
        raise NotImplementedError()
