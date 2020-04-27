from arekit.common.embeddings.base import Embedding
from arekit.networks.embedding.entity_masks import EntityMasks


def __entity_mask_to_word(mask):
    if mask == EntityMasks.ANY_ENTITY_MASK:
        return u"e"
    elif mask == EntityMasks.OBJ_ENTITY_MASK:
        return u"object"
    elif mask == EntityMasks.SUBJ_ENTITY_MASK:
        return u"subject"

    return None


def generate_entity_embeddings(use_types, word_embedding):
    assert(isinstance(use_types, bool))
    assert(isinstance(word_embedding, Embedding))

    # Unique start index
    embeddings = {}

    for e_mask in EntityMasks.iter_supported_entity_masks():
        value = EntityMasks.compose(e_mask=e_mask, e_type=None)

        if value not in embeddings:
            mask = __entity_mask_to_word(e_mask)
            m_ind = word_embedding.try_find_index_by_plain_word(mask)
            embeddings[value] = word_embedding.get_vector_by_index(m_ind)

    return embeddings

