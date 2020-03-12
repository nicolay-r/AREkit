ENTITY_TYPE_SEPARATOR = u'_'


class EntityMasks:

    ANY_ENTITY_MASK = u"ENTITY"
    OBJ_ENTITY_MASK = u"E_OBJ"
    SUBJ_ENTITY_MASK = u"E_SUBJ"

    @staticmethod
    def compose(e_mask, e_type):
        assert(isinstance(e_mask, unicode))
        assert(isinstance(e_type, unicode) or e_type is None)
        if e_type is not None:
            return u'{}{}{}'.format(e_mask, ENTITY_TYPE_SEPARATOR, e_type)
        else:
            return u"{}".format(e_mask)

    @staticmethod
    def iter_supported_entity_masks():
        yield EntityMasks.ANY_ENTITY_MASK
        yield EntityMasks.OBJ_ENTITY_MASK
        yield EntityMasks.SUBJ_ENTITY_MASK


def iter_entity_masks():
    for entity_mask in EntityMasks.iter_supported_entity_masks():
        yield entity_mask
