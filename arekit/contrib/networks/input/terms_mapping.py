from arekit.common.data.input.terms_mapper import OpinionContainingTextTermsMapper
from arekit.common.entities.base import Entity
from arekit.common.frames.text_variant import TextFrameVariant
from arekit.contrib.networks.input.term_types import TermTypes


class VectorizedNetworkTermMapping(OpinionContainingTextTermsMapper):
    """ For every element returns: (word, embedded vector)
    """

    def __init__(self, string_entities_formatter, vectorizers):
        """string_emb_entity_formatter:
            Utilized in order to obtain embedding value from predefined_embeding for entities
           vectorizers:
            dict
        """
        assert(isinstance(vectorizers, dict))

        for term_type in TermTypes.iter_types():
            assert(term_type in vectorizers)

        super(VectorizedNetworkTermMapping, self).__init__(
            entity_formatter=string_entities_formatter)

        self.__vectorizers = vectorizers

    def map_term(self, term_type, term):
        """Universal term mapping method.

        Args:
            term_type (TermTypes): The type of term to map.
            term (str): The term to map.

        Returns:
            The mapped term.
        """
        return self.__vectorizers[term_type].create_term_embedding(term=term)

    def map_word(self, w_ind, word):
        return self.map_term(TermTypes.WORD, word)

    def map_text_frame_variant(self, fv_ind, text_frame_variant):
        assert(isinstance(text_frame_variant, TextFrameVariant))
        return self.map_term(TermTypes.FRAME, text_frame_variant.Variant.get_value())

    def map_token(self, t_ind, token):
        """ It assumes to be composed for all the supported types.
        """
        return self.map_term(TermTypes.TOKEN, token.get_token_value())

    def map_entity(self, e_ind, entity):
        assert(isinstance(entity, Entity))

        # Value extraction
        str_formatted_entity = super(VectorizedNetworkTermMapping, self).map_entity(
            e_ind=e_ind,
            entity=entity)

        # Vector extraction
        return self.map_term(TermTypes.ENTITY, str_formatted_entity)
