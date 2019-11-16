import collections

import numpy as np

from core.common.entities.entity import Entity
from core.common.text_frame_variant import TextFrameVariant
from core.networks.context.embedding import entity
from core.networks.context.embedding.offsets import TermsEmbeddingOffsets
from core.processing.pos.base import POSTagger
from core.processing.text.token import Token
from core.processing.text.tokens import Tokens
from core.common.embeddings.embedding import Embedding
from core.common.embeddings.tokens import TokenEmbedding
from core.networks.context.debug import DebugKeys


def calculate_embedding_indices_for_terms(terms,
                                          term_embedding_matrix,
                                          word_embedding,
                                          syn_subj_indices,
                                          syn_obj_indices,
                                          missed_word_embedding,
                                          token_embedding,
                                          frames_embedding):
    assert(isinstance(terms, collections.Iterable))
    assert(isinstance(term_embedding_matrix, np.ndarray))
    assert(isinstance(word_embedding, Embedding))
    assert(isinstance(syn_obj_indices, set))
    assert(isinstance(syn_subj_indices, set))
    assert(isinstance(missed_word_embedding, Embedding))
    assert(isinstance(token_embedding, TokenEmbedding))
    assert(isinstance(frames_embedding, Embedding))

    indices = []
    embedding_offsets = TermsEmbeddingOffsets(words_count=word_embedding.VocabularySize,
                                              missed_words_count=missed_word_embedding.VocabularySize,
                                              tokens_count=token_embedding.VocabularySize,
                                              frames_count=frames_embedding.VocabularySize)

    unknown_word_embedding_index = embedding_offsets.get_token_index(
        token_embedding.find_index_by_word(Tokens.UNKNOWN_WORD))

    debug_words_found = 0
    debug_words_count = 0
    for i, term in enumerate(terms):
        if isinstance(term, unicode):
            index = unknown_word_embedding_index
            if term in word_embedding:
                index = embedding_offsets.get_word_index(word_embedding.find_index_by_word(term))
            elif term in missed_word_embedding:
                index = embedding_offsets.get_missed_word_index(missed_word_embedding.find_index_by_word(term))
                debug_words_found += int(term in word_embedding)
                debug_words_count += 1
        elif isinstance(term, Token):
            index = embedding_offsets.get_token_index(token_embedding.find_index_by_word(term.get_token_value()))
        elif isinstance(term, TextFrameVariant):
            index = embedding_offsets.get_frame_index(frames_embedding.find_index_by_word(term.Variant.get_value()))
        elif isinstance(term, Entity):
            e_mask = __select_enity_mask(index=i,
                                         subjs_set=syn_subj_indices,
                                         objs_set=syn_obj_indices)
            e_value = entity.compose_entity_mask(e_mask=e_mask,
                                                 e_type=term.Type)
            index = embedding_offsets.get_missed_word_index(missed_word_embedding.find_index_by_word(e_value))
        else:
            raise Exception("Unsuported type {}".format(term))

        indices.append(index)

    if DebugKeys.EmbeddingIndicesPercentWordsFound:
        print "Words found: {} ({}%)".format(debug_words_found,
                                             100.0 * debug_words_found / debug_words_count)
        print "Words missed: {} ({}%)".format(debug_words_count - debug_words_found,
                                              100.0 * (debug_words_count - debug_words_found) / debug_words_count)

    return indices


def __select_enity_mask(index, subjs_set, objs_set):
    if index in objs_set:
        return entity.OBJ_ENTITY_MASK
    elif index in subjs_set:
        return entity.SUBJ_ENTITY_MASK
    return entity.ANY_ENTITY_MASK


def calculate_pos_indices_for_terms(terms, pos_tagger):
    assert(isinstance(terms, collections.Iterable))
    assert(isinstance(pos_tagger, POSTagger))

    indices = []

    for index, term in enumerate(terms):
        if isinstance(term, Token):
            pos = pos_tagger.Empty
        elif isinstance(term, unicode):
            pos = pos_tagger.get_term_pos(term)
        else:
            pos = pos_tagger.Unknown

        indices.append(pos_tagger.pos_to_int(pos))

    return indices
