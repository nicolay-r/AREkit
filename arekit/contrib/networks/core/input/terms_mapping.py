from arekit.common.data.input.terms_mapper import OpinionContainingTextTermsMapper
from arekit.common.entities.base import Entity
from arekit.common.frames.text_variant import TextFrameVariant


class StringWithEmbeddingNetworkTermMapping(OpinionContainingTextTermsMapper):
    """ For every element returns: (word, embedded vector)
    """

    WORD = "word"
    ENTITY = "entity"
    FRAME = "frame"
    TOKEN = "token"

    def __init__(self, string_entities_formatter, vectorizers):
        """
        string_emb_entity_formatter:
            Utilized in order to obtain embedding value from predefined_embeding for enties
        vectorizers:
            dict
        """
        assert(isinstance(vectorizers, dict))
        assert(self.WORD in vectorizers)
        assert(self.ENTITY in vectorizers)
        assert(self.FRAME in vectorizers)
        assert(self.TOKEN in vectorizers)

        super(StringWithEmbeddingNetworkTermMapping, self).__init__(
            entity_formatter=string_entities_formatter)

        self.__vectorizers = vectorizers

    def map_word(self, w_ind, word):
        value, vector = self.__vectorizers[self.WORD].create_term_embedding(term=word)
        return value, vector

    def map_text_frame_variant(self, fv_ind, text_frame_variant):
        assert(isinstance(text_frame_variant, TextFrameVariant))
        value, vector = self.__vectorizers[self.FRAME].create_term_embedding(
            term=text_frame_variant.Variant.get_value())
        return value, vector

    def map_token(self, t_ind, token):
        """ It assumes to be composed for all the supported types.
        """
        value = token.get_token_value()
        vector = self.__vectorizers[self.TOKEN].create_term_embedding(term=t_ind)
        return value, vector

    def map_entity(self, e_ind, entity):
        assert(isinstance(entity, Entity))

        # Value extraction
        str_formatted_entity = super(StringWithEmbeddingNetworkTermMapping, self).map_entity(
            e_ind=e_ind,
            entity=entity)

        # Vector extraction
        emb_word, vector = self.__vectorizers[self.ENTITY].create_term_embedding(term=str_formatted_entity)

        return emb_word, vector
