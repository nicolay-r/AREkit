from arekit.common.context.terms_mapper import TextTermsMapper
from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import EntityType
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.processing.text.token import Token


class OpinionContainingTextTermsMapper(TextTermsMapper):
    """
    Provides an ability to setup s_obj, t_obj
    The latter might be utilized with synonyms collection
    """

    def __init__(self, entity_formatter, synonyms):
        assert(isinstance(entity_formatter, StringEntitiesFormatter))
        assert(isinstance(synonyms, SynonymsCollection) or synonyms is None)
        self.__entities_formatter = entity_formatter
        self.__synonyms = synonyms
        self.__s_ind = None
        self.__t_ind = None
        self.__s_group = None
        self.__t_group = None

    @property
    def StringEntitiesFormatter(self):
        return self.__entities_formatter

    def __syn_group(self, entity):
        assert(isinstance(entity, Entity))

        if entity is None:
            return None

        if self.__synonyms is None:
            return None

        if not self.__synonyms.contains_synonym_value(entity.Value):
            return None

        return self.__synonyms.get_synonym_group_index(entity.Value)

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

        for mapped in super(OpinionContainingTextTermsMapper, self).iter_mapped(terms):
            yield mapped

    def map_entity(self, e_ind, entity):
        if e_ind == self.__s_ind:
            return self.__entities_formatter.to_string(original_value=entity,
                                                       entity_type=EntityType.Subject)
        elif e_ind == self.__t_ind:
            return self.__entities_formatter.to_string(original_value=entity,
                                                       entity_type=EntityType.Object)
        elif self.__is_in_same_group(self.__syn_group(entity), self.__s_group):
            return self.__entities_formatter.to_string(original_value=entity,
                                                       entity_type=EntityType.SynonymSubject)
        elif self.__is_in_same_group(self.__syn_group(entity), self.__t_group):
            return self.__entities_formatter.to_string(original_value=entity,
                                                       entity_type=EntityType.SynonymObject)
        else:
            return self.__entities_formatter.to_string(original_value=entity,
                                                       entity_type=EntityType.Other)

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
