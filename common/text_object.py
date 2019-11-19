from core.common.bound import Bound


class TextObject(object):
    """
    Considering any part of text, labeled by 'position', and 'type'
    The latter is used to emphasize the entity type.
    """

    def __init__(self, terms, obj_type, position):
        assert(isinstance(terms, list))
        assert(isinstance(position, int))
        assert(isinstance(obj_type, unicode) or obj_type is None)
        self.__terms = terms
        self.__position = position
        self.__type = obj_type
        self.__tag = None

    # region properties

    @property
    def Position(self):
        return self.__position

    @property
    def Tag(self):
        return self.__tag

    @property
    def Type(self):
        return self.__type

    # endregion

    # region public methods

    def set_tag(self, value):
        self.__tag = value

    def get_value(self):
        return u' '.join(self.__terms)

    def get_bound(self):
        return Bound(self.__position, len(self.__terms))

    def iter_terms(self):
        for term in self.__terms:
            yield term

    # endregion

    # region overriden

    def __len__(self):
        return len(self.__terms)

    # endregion
