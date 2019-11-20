ANY_ENTITY_MASK = u"ENTITY"
OBJ_ENTITY_MASK = u"E_OBJ"
SUBJ_ENTITY_MASK = u"E_SUBJ"
# TODO. Add NO_ENTITY_MASK = u"NO"

ENTITY_TYPE_SEPARATOR = u'_'

__supported_entity_masks = [ANY_ENTITY_MASK, OBJ_ENTITY_MASK, SUBJ_ENTITY_MASK]
__supported_entity_types = [u'PER', u'LOC', u'ORG', u'GEOPOLIT']
# TODO. Add no


def iter_entity_types():
    for entity_type in __supported_entity_types:
        yield entity_type


def iter_entity_masks():
    for entity_mask in __supported_entity_masks:
        yield entity_mask


# TODO e_type=None
def compose_entity_mask(e_mask, e_type):
    assert(isinstance(e_mask, unicode))
    assert(isinstance(e_type, unicode))
    return u'{}{}{}'.format(e_mask, ENTITY_TYPE_SEPARATOR, e_type)