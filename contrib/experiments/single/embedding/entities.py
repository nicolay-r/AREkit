import itertools
import numpy as np

from arekit.common.embeddings.base import Embedding
from arekit.common.synonyms import SynonymsCollection
from arekit.networks.context.embedding.entity import EntityMasks


################################
# Extra entity types
################################

CAPITAL_ENTITY_TYPE = u"CAPITAL"
STATE_ENTITY_TYPE = u"STATE"
PER_ENTITY_TYPE = u'PER'
LOC_ENTITY_TYPE = u'LOC'
ORG_ENTITY_TYPE = u'ORG'
GEOPOLIT_ENTITY_TYPE = u'GEOPOLIT'


def provide_entity_type_by_value(
        synonyms,
        value,
        states_set,
        capitals_set):
    assert(isinstance(synonyms, SynonymsCollection))

    if not synonyms.contains_synonym_value(value):
        return None

    for s_value in synonyms.iter_synonym_values(value):
        if s_value in capitals_set:
            return CAPITAL_ENTITY_TYPE

    for s_value in synonyms.iter_synonym_values(value):
        if s_value in states_set:
            return STATE_ENTITY_TYPE

    return None


def __entity_mask_to_word(mask):
    if mask == EntityMasks.ANY_ENTITY_MASK:
        return u"e"
    elif mask == EntityMasks.OBJ_ENTITY_MASK:
        return u"object"
    elif mask == EntityMasks.SUBJ_ENTITY_MASK:
        return u"subject"

    return None


def __entity_type_to_word(e_type):
    if e_type == ORG_ENTITY_TYPE:
        return u"organization"
    if e_type == LOC_ENTITY_TYPE:
        return u'location'
    if e_type == PER_ENTITY_TYPE:
        return u'person'
    if e_type == GEOPOLIT_ENTITY_TYPE:
        return u'political'
    if e_type == CAPITAL_ENTITY_TYPE:
        return u'capital'
    if e_type == STATE_ENTITY_TYPE:
        return u'state'


# TODO. External
def iter_all_entity_types():
    return itertools.chain(iter_entity_types(),
                           [CAPITAL_ENTITY_TYPE,
                            STATE_ENTITY_TYPE])


def generate_entity_embeddings(use_types, word_embedding):
    assert(isinstance(use_types, bool))
    assert(isinstance(word_embedding, Embedding))

    # Unique start index
    embeddings = {}

    v_index = 0

    for e_mask in EntityMasks.iter_supported_entity_masks():
        for e_type in iter_all_entity_types():

            value = EntityMasks.compose(
                e_mask=e_mask,
                e_type=e_type if use_types else None)

            if value not in embeddings:

                mask = __entity_mask_to_word(e_mask)
                t = __entity_type_to_word(e_type)

                m_ind = word_embedding.try_find_index_by_plain_word(mask)
                t_ind = word_embedding.try_find_index_by_plain_word(t)

                e_v = np.mean([word_embedding.get_vector_by_index(m_ind),
                               word_embedding.get_vector_by_index(t_ind)],
                              axis=0)

                embeddings[value] = e_v

            v_index += 1

    return embeddings


def iter_entity_types():
    for entity_type in __supported_entity_types:
        yield entity_type


__supported_entity_types = [PER_ENTITY_TYPE,
                            LOC_ENTITY_TYPE,
                            ORG_ENTITY_TYPE,
                            GEOPOLIT_ENTITY_TYPE]