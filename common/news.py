
class News(object):

    def __init__(self):
        pass

    def iter_linked_text_opinions(self, opinions):
        """
        opinions: iterable Opinion
            is an iterable opinions that should be used to find a related text_opinion entries.
        """
        raise NotImplementedError()
