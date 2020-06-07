import logging
import numpy as np

from arekit.common.context.terms_mapper import TextTermsMapper
from arekit.common.embeddings.base import Embedding
from arekit.common.embeddings.tokens import TokenEmbedding
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.networks.debug import DebugKeys
from arekit.networks.embedding.entity_masks import EntityMasks
from arekit.networks.embedding.offsets import TermsEmbeddingOffsets
from arekit.processing.text.token import Token
from arekit.processing.text.tokens import Tokens

logger = logging.getLogger(__name__)


class IndicingTextTermsMapper(TextTermsMapper):

    # region ctor

    def __init__(self,
                 term_embedding_matrix,
                 word_embedding,
                 syn_subj_indices,
                 syn_obj_indices,
                 custom_word_embedding,
                 token_embedding,
                 frames_embedding,
                 use_entity_types):
        assert(isinstance(term_embedding_matrix, np.ndarray))
        assert(isinstance(word_embedding, Embedding))
        assert(isinstance(syn_obj_indices, set))
        assert(isinstance(syn_subj_indices, set))
        assert(isinstance(custom_word_embedding, Embedding))
        assert(isinstance(token_embedding, TokenEmbedding))
        assert(isinstance(frames_embedding, Embedding))

        self.__word_embedding = word_embedding
        self.__syn_subj_indices = syn_subj_indices
        self.__syn_obj_indices = syn_obj_indices
        self.__custom_word_embedding = custom_word_embedding
        self.__token_embedding = token_embedding
        self.__frames_embedding = frames_embedding
        self.__use_entity_types = use_entity_types

        self.__embedding_offsets = TermsEmbeddingOffsets(
            words_count=word_embedding.VocabularySize,
            custom_words_count=custom_word_embedding.VocabularySize,
            tokens_count=token_embedding.VocabularySize,
            frames_count=frames_embedding.VocabularySize)

        self.__unknown_word_embedding_index = self.__embedding_offsets.get_token_index(
            token_embedding.try_find_index_by_word(Tokens.UNKNOWN_WORD))

        self.__debug_word_found = None
        self.__debug_word_count = None

    # endregion

    # region protected methods

    def _before_mapping(self):
        self.__debug_words_found = 0
        self.__debug_words_count = 0

    def _after_mapping(self):
        if DebugKeys.EmbeddingIndicesPercentWordsFound:

            logger.info("Words found: {} ({}%)".format(
                self.__debug_words_found,
                100.0 * self.__debug_words_found / self.__debug_words_count))

            logger.info("Words custom: {} ({}%)".format(
                self.__debug_words_count - self.__debug_words_found,
                100.0 * (self.__debug_words_count - self.__debug_words_found) / self.__debug_words_count))

    # endregion

    def map_word(self, w_ind, word):
        assert(isinstance(word, unicode))

        index = self.__unknown_word_embedding_index

        if word in self.__word_embedding:
            index = self.__embedding_offsets.get_word_index(self.__word_embedding.try_find_index_by_word(word))
        elif word in self.__custom_word_embedding:
            index = self.__embedding_offsets.get_custom_word_index(self.__custom_word_embedding.try_find_index_by_word(word))
            self.__debug_words_found += int(word in self.__word_embedding)
            self.__debug_words_count += 1

        return index

    def map_token(self, t_ind, token):
        assert(isinstance(token, Token))
        return self.__embedding_offsets.get_token_index(
            self.__token_embedding.try_find_index_by_word(token.get_token_value()))

    def map_text_frame_variant(self, fv_ind, frame_variant):
        assert(isinstance(frame_variant, TextFrameVariant))
        return self.__embedding_offsets.get_frame_index(
            self.__frames_embedding.try_find_index_by_word(frame_variant.Variant.get_value()))

    def map_entity(self, e_ind, entity):
        e_mask = EntityMasks.select_mask(index=e_ind,
                                         subjects_set=self.__syn_subj_indices,
                                         objects_set=self.__syn_obj_indices)
        e_value = EntityMasks.compose(e_mask=e_mask,
                                      e_type=entity.Type if self.__use_entity_types else None)
        return self.__embedding_offsets.get_custom_word_index(
            self.__custom_word_embedding.try_find_index_by_word(e_value))