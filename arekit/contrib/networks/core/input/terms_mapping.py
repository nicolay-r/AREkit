import numpy as np

from arekit.common.data.input.terms_mapper import OpinionContainingTextTermsMapper
from arekit.common.entities.base import Entity
from arekit.common.frames.text_variant import TextFrameVariant
from arekit.contrib.networks.core.input.embedding.custom import create_term_embedding
from arekit.contrib.networks.embeddings.base import Embedding


class StringWithEmbeddingNetworkTermMapping(OpinionContainingTextTermsMapper):
    """
    For every element returns: (word, embedded vector)
    """

    MAX_PART_CUSTOM_EMBEDDING_SIZE = 3
    TOKEN_RANDOM_SEED_OFFSET = 12345

    def __init__(self, predefined_embedding, string_entities_formatter):
        """
        predefined_embedding:
        string_emb_entity_formatter:
            Utilized in order to obtain embedding value from predefined_embeding for enties
        """
        assert(isinstance(predefined_embedding, Embedding))

        super(StringWithEmbeddingNetworkTermMapping, self).__init__(
            entity_formatter=string_entities_formatter)

        self.__predefined_embedding = predefined_embedding

    def map_word(self, w_ind, word):
        value, vector = create_term_embedding(term=word,
                                              embedding=self.__predefined_embedding,
                                              max_part_size=self.MAX_PART_CUSTOM_EMBEDDING_SIZE)
        return value, vector

    def map_text_frame_variant(self, fv_ind, text_frame_variant):
        assert(isinstance(text_frame_variant, TextFrameVariant))
        value, embedding = create_term_embedding(term=text_frame_variant.Variant.get_value(),
                                                 embedding=self.__predefined_embedding,
                                                 max_part_size=self.MAX_PART_CUSTOM_EMBEDDING_SIZE)

        return value, embedding

    def map_token(self, t_ind, token):
        """
        It assumes to be composed for all the supported types.
        """
        value = token.get_token_value()

        seed_token_offset = self.TOKEN_RANDOM_SEED_OFFSET

        # TODO. #348 related. Move it into `utils` contrib.
        vector = self.__get_random_normal_distribution(
            vector_size=self.__predefined_embedding.VectorSize,
            seed=t_ind + seed_token_offset,
            loc=0.05,
            scale=0.025)

        return value, vector

    def map_entity(self, e_ind, entity):
        """
        Since entities are masked, it assumes to be composed for all the supported types.
        """
        assert(isinstance(entity, Entity))

        # Value extraction
        str_formatted_entity = super(StringWithEmbeddingNetworkTermMapping, self).map_entity(
            e_ind=e_ind,
            entity=entity)

        # Vector extraction
        emb_word, vector = create_term_embedding(term=str_formatted_entity,
                                                 max_part_size=self.MAX_PART_CUSTOM_EMBEDDING_SIZE,
                                                 embedding=self.__predefined_embedding)

        return emb_word, vector

    # region private methods

    @staticmethod
    def __get_random_normal_distribution(vector_size, seed, loc, scale):
        assert (isinstance(vector_size, int))
        assert (isinstance(seed, int))
        np.random.seed(seed)
        return np.random.normal(loc=loc, scale=scale, size=vector_size)

    # endregion
