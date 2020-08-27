from arekit.common.bound import Bound
from arekit.common.entities.base import Entity


class TextObject(object):
    """
    Considering any part of text, labeled by 'position', and 'type'
    The latter is used to emphasize the entity type.
    """

    def __init__(self, id_in_sentence, value, obj_type, position, terms_count, syn_group_index):
        assert(isinstance(id_in_sentence, int))
        assert(isinstance(value, unicode))
        assert(isinstance(position, int))
        assert(isinstance(terms_count, int) and terms_count > 0)
        assert(isinstance(obj_type, unicode) or obj_type is None)
        assert(isinstance(syn_group_index, int))
        self.__value = value
        self.__type = obj_type
        self.__id_in_sentence = id_in_sentence
        self.__syn_group_index = syn_group_index
        self.__bound = Bound(pos=position, length=terms_count)

    def to_entity(self, to_doc_id_func):
        assert(callable(to_doc_id_func))
        return Entity(value=self.__value if len(self.__value) > 0 else u'[empty]',
                      e_type=self.Type,
                      id_in_doc=to_doc_id_func(self.IdInSentence))

    # region properties

    @property
    def Value(self):
        return self.__value

    @property
    def Type(self):
        return self.__type

    @property
    def IdInSentence(self):
        return self.__id_in_sentence

    @property
    def Bound(self):
        return self.__bound

    # endregion

    # region overriden

    def __len__(self):
        return len(self.__terms)

    # endregion
