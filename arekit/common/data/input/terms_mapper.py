from arekit.common.context.terms_mapper import TextTermsMapper
from arekit.common.context.token import Token
from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import OpinionEntityType
from arekit.common.frames.text_variant import TextFrameVariant


class OpinionContainingTextTermsMapper(TextTermsMapper):
    """
    Provides an ability to setup s_obj, t_obj
    The latter might be utilized with synonyms collection
    """

    def __init__(self, entity_formatter):
        assert(isinstance(entity_formatter, StringEntitiesFormatter))
        self.__entities_formatter = entity_formatter
        self.__s_ind = None
        self.__t_ind = None
        self.__s_group = None
        self.__t_group = None

    @property
    def StringEntitiesFormatter(self):
        return self.__entities_formatter

    def __syn_group(self, entity):
        """ Note: here we guarantee that entity has GroupIndex.
        """
        assert(isinstance(entity, Entity))
        return entity.GroupIndex if entity is not None else None

    def set_s_ind(self, s_ind):
        assert(isinstance(s_ind, int))
        self.__s_ind = s_ind

    def set_t_ind(self, t_ind):
        assert(isinstance(t_ind, int))
        self.__t_ind = t_ind

    def _after_mapping(self):
        """ In order to prevent bugs.
            Every index should be declared before mapping.
        """
        self.__s_ind = None
        self.__t_ind = None

    def iter_mapped(self, terms):
        terms_list = list(terms)
        self.__s_group = self.__syn_group(terms_list[self.__s_ind] if self.__s_ind is not None else None)
        self.__t_group = self.__syn_group(terms_list[self.__t_ind] if self.__t_ind is not None else None)
        return super(OpinionContainingTextTermsMapper, self).iter_mapped(terms)

    def map_entity(self, e_ind, entity):

        entity_type = OpinionEntityType.Other
        if e_ind == self.__s_ind:
            entity_type = OpinionEntityType.Subject
        elif e_ind == self.__t_ind:
            entity_type = OpinionEntityType.Object
        elif self.__is_in_same_group(self.__syn_group(entity), self.__s_group):
            entity_type = OpinionEntityType.SynonymSubject
        elif self.__is_in_same_group(self.__syn_group(entity), self.__t_group):
            entity_type = OpinionEntityType.SynonymObject

        return self.__entities_formatter.to_string(original_value=entity,
                                                   entity_type=entity_type)

    @staticmethod
    def __is_in_same_group(g1, g2):

        if g1 is None or g2 is None:
            # In such scenario we cannot guarantee
            # that g1 and g2 belong to the same group.
            return False

        return g1 == g2

    def map_word(self, w_ind, word):
        return word.strip()

    def map_text_frame_variant(self, fv_ind, text_frame_variant):
        assert(isinstance(text_frame_variant, TextFrameVariant))
        return text_frame_variant.Variant.get_value().strip()

    def map_token(self, t_ind, token):
        assert(isinstance(token, Token))
        return token.get_meta_value()
