ANY_ENTITY_MASK = u"ENTITY"
OBJ_ENTITY_MASK = u"E_OBJ"
SUBJ_ENTITY_MASK = u"E_SUBJ"

ENTITY_TYPE_SEPARATOR = u'_'

PER_ENTITY_TYPE = u'PER'
LOC_ENTITY_TYPE = u'LOC'
ORG_ENTITY_TYPE = u'ORG'
GEOPOLIT_ENTITY_TYPE = u'GEOPOLIT'

__supported_entity_masks = [ANY_ENTITY_MASK,
                            OBJ_ENTITY_MASK,
                            SUBJ_ENTITY_MASK]

__supported_entity_types = [PER_ENTITY_TYPE,
                            LOC_ENTITY_TYPE,
                            ORG_ENTITY_TYPE,
                            GEOPOLIT_ENTITY_TYPE]


def iter_entity_types():
    for entity_type in __supported_entity_types:
        yield entity_type


def iter_entity_masks():
    for entity_mask in __supported_entity_masks:
        yield entity_mask


def compose_entity_mask(e_mask, e_type):
    assert(isinstance(e_mask, unicode))
    assert(isinstance(e_type, unicode) or e_type is None)
    if e_type is not None:
        return u'{}{}{}'.format(e_mask, ENTITY_TYPE_SEPARATOR, e_type)
    else:
        return u"{}".format(e_mask)
