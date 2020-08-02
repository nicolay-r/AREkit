from core.common.bound import Bound


class TextObject(object):

    def __init__(self, terms, position):
        assert(isinstance(terms, list))
        assert(isinstance(position, int))
        self.__terms = terms
        self.__position = position
        self.__tag = None

    @property
    def Position(self):
        return self.__position

    @property
    def Tag(self):
        return self.__tag

    def set_tag(self, value):
        self.__tag = value

    def get_value(self):
        return ' '.join(self.__terms)

    def get_bound(self):
        return Bound(self.__position, len(self.__terms))

    def iter_terms(self):
        for term in self.__terms:
            yield term

    def __len__(self):
        return len(self.__terms)
