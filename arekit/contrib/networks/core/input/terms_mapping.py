import numpy as np

from arekit.common.data.input.terms_mapper import OpinionContainingTextTermsMapper
from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import EntityType
from arekit.common.frames.text_variant import TextFrameVariant
from arekit.contrib.networks.core.input.embedding.custom import create_term_embedding
from arekit.contrib.networks.embeddings.base import Embedding


class StringWithEmbeddingNetworkTermMapping(OpinionContainingTextTermsMapper):
    """
    For every element returns: (word, embedded vector)
    """

    MAX_PART_CUSTOM_EMBEDDING_SIZE = 3
    TOKEN_RANDOM_SEED_OFFSET = 12345

    def __init__(self, predefined_embedding, string_entities_formatter, string_emb_entity_formatter):
        """
        predefined_embedding:
        string_emb_entity_formatter:
            Utilized in order to obtain embedding value from predefined_embeding for enties
        """
        assert(isinstance(predefined_embedding, Embedding))
        assert(isinstance(string_emb_entity_formatter, StringEntitiesFormatter))

        super(StringWithEmbeddingNetworkTermMapping, self).__init__(
            entity_formatter=string_entities_formatter)

        self.__predefined_embedding = predefined_embedding
        self.__string_emb_entity_formatter = string_emb_entity_formatter

    def map_word(self, w_ind, word):
        value, vector = create_term_embedding(term=word,
                                              embedding=self.__predefined_embedding,
                                              check=True)
        return value, vector

    def map_text_frame_variant(self, fv_ind, text_frame_variant):
        assert(isinstance(text_frame_variant, TextFrameVariant))
        value, embedding = create_term_embedding(term=text_frame_variant.Variant.get_value(),
                                                 embedding=self.__predefined_embedding,
                                                 check=False)

        return value, embedding

    def map_token(self, t_ind, token):
        """
        It assumes to be composed for all the supported types.
        """
        value = token.get_token_value()

        seed_token_offset = self.TOKEN_RANDOM_SEED_OFFSET

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
        str_entity_mask = super(StringWithEmbeddingNetworkTermMapping, self).map_entity(
            e_ind=e_ind,
            entity=entity)

        # TODO. Use synonyms. (Or use this functionality in a base class).
        empty_set = set()
        e_type = self.__get_entity_type(e_ind=e_ind,
                                        subj_ind_set=empty_set,
                                        obj_ind_set=empty_set)

        # Vector extraction
        entity_word = self.__string_emb_entity_formatter.to_string(original_value=None,
                                                                   entity_type=e_type)
        m_ind = self.__predefined_embedding.try_find_index_by_plain_word(entity_word)
        vector = self.__predefined_embedding.get_vector_by_index(m_ind)

        return str_entity_mask, vector

    # region private methods

    @staticmethod
    def __get_random_normal_distribution(vector_size, seed, loc, scale):
        assert (isinstance(vector_size, int))
        assert (isinstance(seed, int))
        np.random.seed(seed)
        return np.random.normal(loc=loc, scale=scale, size=vector_size)

    @staticmethod
    def __get_entity_type(e_ind, subj_ind_set, obj_ind_set):
        assert(isinstance(e_ind, int))
        assert(isinstance(subj_ind_set, set))
        assert(isinstance(obj_ind_set, set))

        result = EntityType.Other
        if e_ind in obj_ind_set:
            result = EntityType.Object
        elif e_ind in subj_ind_set:
            result = EntityType.Subject

        return result

    # endregion