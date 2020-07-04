from arekit.common.context.terms_mapper import TextTermsMapper
from arekit.common.embeddings.base import Embedding
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.networks.tf_models.single.embedding.custom import create_term_embedding


class EmbeddedTermMapping(TextTermsMapper):
    """
    For every element returns: (word, embedded vector)
    """

    MAX_PART_CUSTOM_EMBEDDING_SIZE = 3

    def __init__(self, predefined_embedding):
        assert(isinstance(predefined_embedding, Embedding))
        self.__predefined_embedding = predefined_embedding

    def map_word(self, w_ind, word):
        if word in self.__predefined_embedding:
            vector = self.__predefined_embedding[word]
        else:
            vector = create_term_embedding(term=word,
                                           embedding=self.__predefined_embedding,
                                           max_part_size=self.MAX_PART_CUSTOM_EMBEDDING_SIZE)

        return word, vector

    def map_text_frame_variant(self, fv_ind, text_frame_variant):
        assert(isinstance(text_frame_variant, TextFrameVariant))
        variant_value = text_frame_variant.Variant.get_value()
        embedding = create_term_embedding(term=variant_value,
                                          embedding=self.__predefined_embedding,
                                          max_part_size=self.MAX_PART_CUSTOM_EMBEDDING_SIZE)

        return variant_value, embedding

    def map_token(self, t_ind, token):
        """
        It assumes to be composed for all the supported types.
        """
        # TODO. Here we need to return values for embedding!
        # TODO. Decide token string value.
        raise NotImplementedError()

    def map_entity(self, e_ind, entity):
        """
        Since entities are masked, it assumes to be composed for all the supported types.
        """
        # TODO. Here we need to return values for embedding!
        # TODO. Decide entity string value.
        raise NotImplementedError()

