from arekit.common.bound import Bound
from arekit.common.entities.base import Entity


class TextObject(object):
    """
    Considering any part of text, labeled by 'position', and 'type'
    The latter is used to emphasize the entity type.
    """

    def __init__(self, id_in_sentence, terms, obj_type, position):
        assert(isinstance(id_in_sentence, int))
        assert(isinstance(terms, list))
        assert(isinstance(position, int))
        assert(isinstance(obj_type, unicode) or obj_type is None)
        self.__terms = terms
        self.__position = position
        self.__type = obj_type
        self.__id_in_sentence = id_in_sentence
        self.__tag = None

    def to_entity(self, to_doc_id_func):
        assert(callable(to_doc_id_func))

        _value = self.get_value()
        value = _value if len(_value) > 0 else u'[empty]'

        return Entity(value=value,
                      e_type=self.Type,
                      id_in_doc=to_doc_id_func(self.IdInSentence))

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

    @property
    def IdInSentence(self):
        return self.__id_in_sentence

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
