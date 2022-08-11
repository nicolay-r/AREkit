from arekit.common.bound import Bound
from arekit.contrib.source.brat.entities.entity import BratEntity


class TextObject(object):
    """
    Considering any part of text, labeled by 'position', and 'type'
    The latter is used to emphasize the entity type.
    """

    def __init__(self, id_in_sentence, value, obj_type, position, terms_count, syn_group_index, is_auth):
        assert(isinstance(id_in_sentence, int))
        assert(isinstance(value, str))
        assert(isinstance(position, int))
        assert(isinstance(terms_count, int) and terms_count > 0)
        assert(isinstance(obj_type, str) or obj_type is None)
        assert(isinstance(syn_group_index, int))
        assert(isinstance(is_auth, bool))
        self.__value = value
        self.__type = obj_type
        self.__id_in_sentence = id_in_sentence
        self.__syn_group_index = syn_group_index
        self.__is_auth = is_auth
        self.__bound = Bound(pos=position, length=terms_count)

    def to_entity(self, to_doc_id_func):
        assert(callable(to_doc_id_func))
        return BratEntity(id_in_doc=to_doc_id_func(self.__id_in_sentence),
                          value=self.__value if len(self.__value) > 0 else '[empty]',
                          e_type=self.__type,
                          index_begin=self.__bound.Position,
                          index_end=self.__bound.Position + self.__bound.Length,
                          group_index=self.__syn_group_index)

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

    @property
    def IsAuthorized(self):
        return self.__is_auth

    # endregion
