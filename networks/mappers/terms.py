import logging
import numpy as np

from arekit.common.context.terms_mapper import TextTermsMapper
from arekit.common.embeddings.base import Embedding
from arekit.common.embeddings.tokens import TokenEmbedding
from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.networks.input.embedding.offsets import TermsEmbeddingOffsets
from arekit.processing.text.token import Token

logger = logging.getLogger(__name__)


# TODO. This will be removed as in Samples we deal with words and a whole vocabulary.
# TODO. This will be removed as in Samples we deal with words and a whole vocabulary.
# TODO. This will be removed as in Samples we deal with words and a whole vocabulary.
class IndexingTextTermsMapper(TextTermsMapper):

    # region ctor

    def __init__(self,
                 term_embedding_matrix,
                 word_embedding,
                 syn_subj_indices,
                 syn_obj_indices,
                 entity_embedding,
                 token_embedding,
                 string_entity_formatter):
        assert(isinstance(term_embedding_matrix, np.ndarray))
        assert(isinstance(word_embedding, Embedding))
        assert(isinstance(syn_obj_indices, set))
        assert(isinstance(syn_subj_indices, set))
        assert(isinstance(entity_embedding, Embedding))
        assert(isinstance(token_embedding, TokenEmbedding))
        assert(isinstance(string_entity_formatter, StringEntitiesFormatter))

        self.__word_embedding = word_embedding
        self.__syn_subj_indices = syn_subj_indices
        self.__syn_obj_indices = syn_obj_indices
        self.__entity_embedding = entity_embedding
        self.__token_embedding = token_embedding
        self.__string_entities_formatter = string_entity_formatter

        self.__embedding_offsets = TermsEmbeddingOffsets(words_count=word_embedding.VocabularySize)

    # endregion

    def map_word(self, w_ind, word):
        assert(isinstance(word, unicode))
        raise NotImplementedError()

    def map_token(self, t_ind, token):
        assert(isinstance(token, Token))
        raise NotImplementedError()

    def map_text_frame_variant(self, fv_ind, frame_variant):
        assert(isinstance(frame_variant, TextFrameVariant))
        return self.__embedding_offsets.get_word_index(self.__word_embedding.try_find_index_by_word(frame_variant.Variant.get_value()))

    def map_entity(self, e_ind, entity):
        assert(isinstance(entity, Entity))
        raise NotImplementedError()
