import collections
import logging

import numpy as np

from arekit.common.entities.base import Entity
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.networks.context.embedding import entity
from arekit.networks.context.embedding.offsets import TermsEmbeddingOffsets
from arekit.processing.pos.base import POSTagger
from arekit.processing.text.token import Token
from arekit.processing.text.tokens import Tokens
from arekit.common.embeddings.base import Embedding
from arekit.common.embeddings.tokens import TokenEmbedding
from arekit.networks.context.debug import DebugKeys


logger = logging.getLogger(__name__)


# region private functions

def __select_enity_mask(index, subjs_set, objs_set):
    if index in objs_set:
        return entity.OBJ_ENTITY_MASK
    elif index in subjs_set:
        return entity.SUBJ_ENTITY_MASK
    return entity.ANY_ENTITY_MASK

# endregion


def iter_embedding_indices_for_terms(terms,
                                     term_embedding_matrix,
                                     word_embedding,
                                     syn_subj_indices,
                                     syn_obj_indices,
                                     custom_word_embedding,
                                     token_embedding,
                                     frames_embedding,
                                     use_entity_types):
    assert(isinstance(terms, collections.Iterable))
    assert(isinstance(term_embedding_matrix, np.ndarray))
    assert(isinstance(word_embedding, Embedding))
    assert(isinstance(syn_obj_indices, set))
    assert(isinstance(syn_subj_indices, set))
    assert(isinstance(custom_word_embedding, Embedding))
    assert(isinstance(token_embedding, TokenEmbedding))
    assert(isinstance(frames_embedding, Embedding))

    embedding_offsets = TermsEmbeddingOffsets(words_count=word_embedding.VocabularySize,
                                              custom_words_count=custom_word_embedding.VocabularySize,
                                              tokens_count=token_embedding.VocabularySize,
                                              frames_count=frames_embedding.VocabularySize)

    unknown_word_embedding_index = embedding_offsets.get_token_index(
        token_embedding.try_find_index_by_word(Tokens.UNKNOWN_WORD))

    debug_words_found = 0
    debug_words_count = 0
    for i, term in enumerate(terms):
        if isinstance(term, unicode):
            index = unknown_word_embedding_index
            if term in word_embedding:
                index = embedding_offsets.get_word_index(word_embedding.try_find_index_by_word(term))
            elif term in custom_word_embedding:
                index = embedding_offsets.get_custom_word_index(custom_word_embedding.try_find_index_by_word(term))
                debug_words_found += int(term in word_embedding)
                debug_words_count += 1
        elif isinstance(term, Token):
            index = embedding_offsets.get_token_index(token_embedding.try_find_index_by_word(term.get_token_value()))
        elif isinstance(term, TextFrameVariant):
            index = embedding_offsets.get_frame_index(frames_embedding.try_find_index_by_word(term.Variant.get_value()))
        elif isinstance(term, Entity):
            e_mask = __select_enity_mask(index=i,
                                         subjs_set=syn_subj_indices,
                                         objs_set=syn_obj_indices)
            e_value = entity.compose_entity_mask(e_mask=e_mask,
                                                 e_type=term.Type if use_entity_types else None)
            index = embedding_offsets.get_custom_word_index(custom_word_embedding.try_find_index_by_word(e_value))
        else:
            raise Exception("Unsuported type {}".format(term))

        yield index

    if DebugKeys.EmbeddingIndicesPercentWordsFound:
        logger.info("Words found: {} ({}%)".format(debug_words_found,
                                             100.0 * debug_words_found / debug_words_count))
        logger.info("Words custom: {} ({}%)".format(debug_words_count - debug_words_found,
                                              100.0 * (debug_words_count - debug_words_found) / debug_words_count))


def iter_pos_indices_for_terms(terms, pos_tagger):
    assert(isinstance(terms, collections.Iterable))
    assert(isinstance(pos_tagger, POSTagger))

    for index, term in enumerate(terms):
        if isinstance(term, Token):
            pos = pos_tagger.Empty
        elif isinstance(term, unicode):
            pos = pos_tagger.get_term_pos(term)
        else:
            pos = pos_tagger.Unknown

        yield pos_tagger.pos_to_int(pos)
