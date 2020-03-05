import itertools
import numpy as np

from arekit.common.embeddings.base import Embedding
from arekit.common.synonyms import SynonymsCollection
from arekit.networks.context.embedding import entity


################################
# Extra entity types
################################
CAPITAL_ENTITY_TYPE = u"CAPITAL"
STATE_ENTITY_TYPE = u"STATE"


def provide_entity_type_by_value(
        synonyms,
        value,
        # TODO. USE set instead due to the performance reason
        states_list,
        # TODO. USE set instead due to the performance reason
        capitals_list):
    assert(isinstance(synonyms, SynonymsCollection))

    if not synonyms.contains_synonym_value(value):
        return None

    for s_value in synonyms.iter_synonym_values(value):
        if s_value in capitals_list:
            # self.__log_capitals_presented += 1
            return CAPITAL_ENTITY_TYPE

    for s_value in synonyms.iter_synonym_values(value):
        if s_value in states_list:
            # self.__log_states_presented += 1
            return STATE_ENTITY_TYPE

    return None


def __entity_mask_to_word(mask):
    if mask == entity.ANY_ENTITY_MASK:
        return u"e"
    elif mask == entity.OBJ_ENTITY_MASK:
        return u"object"
    elif mask == entity.SUBJ_ENTITY_MASK:
        return u"subject"

    return None


def __entity_type_to_word(e_type):
    if e_type == entity.ORG_ENTITY_TYPE:
        return u"organization"
    if e_type == entity.LOC_ENTITY_TYPE:
        return u'location'
    if e_type == entity.PER_ENTITY_TYPE:
        return u'person'
    if e_type == entity.GEOPOLIT_ENTITY_TYPE:
        return u'political'
    if e_type == CAPITAL_ENTITY_TYPE:
        return u'capital'
    if e_type == STATE_ENTITY_TYPE:
        return u'state'


def iter_all_entity_types():
    return itertools.chain(entity.iter_entity_types(),
                           [CAPITAL_ENTITY_TYPE,
                            STATE_ENTITY_TYPE])


def generate_entity_embeddings(use_types, word_embedding):
    assert(isinstance(use_types, bool))
    assert(isinstance(word_embedding, Embedding))

    # Unique start index
    embeddings = {}

    v_index = 0

    for e_mask in entity.iter_entity_masks():
        for e_type in iter_all_entity_types():

            value = entity.compose_entity_mask(
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
