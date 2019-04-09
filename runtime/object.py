from core.common.bound import Bound


class TextObject:
    """
    Result of NER or found by lexicon (adj + noun or noun)
    """

    class ObjectTypes:
        NamedEntity = "NE"
        LexiconUnit = "LEX"

    def __init__(self, terms, position, object_type=None):
        assert(isinstance(terms, list))
        assert(isinstance(position, int))
        self.__terms = terms
        self.__position = position
        self.__obj_type = object_type
        self.__tag = None

    @property
    def Position(self):
        return self.__position

    @property
    def Tag(self):
        return self.__tag

    @property
    def ObjectType(self):
        return self.__obj_type

    @classmethod
    def create_as_named_entity(cls, terms, position):
        return cls(terms, position, object_type=cls.ObjectTypes.NamedEntity)

    @classmethod
    def create_as_lexicon_unit(cls, terms, position):
        return cls(terms, position, object_type=cls.ObjectTypes.LexiconUnit)

    def set_tag(self, value):
        self.__tag = value

    def get_value(self):
        return u' '.join(self.__terms)

    def is_named_entity(self):
        return self.__obj_type == self.ObjectTypes.LexiconUnit

    def get_bound(self):
        return Bound(self.__position, len(self.__terms))

    def iter_terms(self):
        for term in self.__terms:
            yield term

    def __len__(self):
        return len(self.__terms)
