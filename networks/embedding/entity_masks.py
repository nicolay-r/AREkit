# TODO. This should depend on a base str_entity_formatter
class EntityMasks:

    ANY_ENTITY_MASK = u"ENTITY"
    OBJ_ENTITY_MASK = u"E_OBJ"
    SUBJ_ENTITY_MASK = u"E_SUBJ"
    ENTITY_TYPE_SEPARATOR = u'_'

    @staticmethod
    def compose(e_mask, e_type):
        assert(isinstance(e_mask, unicode))
        assert(isinstance(e_type, unicode) or e_type is None)
        if e_type is not None:
            return u'{}{}{}'.format(e_mask, EntityMasks.ENTITY_TYPE_SEPARATOR, e_type)
        else:
            return u"{}".format(e_mask)

    @staticmethod
    def iter_supported_entity_masks():
        yield EntityMasks.ANY_ENTITY_MASK
        yield EntityMasks.OBJ_ENTITY_MASK
        yield EntityMasks.SUBJ_ENTITY_MASK

    @staticmethod
    def select_mask(index, subjects_set, objects_set):
        assert(isinstance(subjects_set, set))
        assert(isinstance(objects_set, set))

        if index in objects_set:
            return EntityMasks.OBJ_ENTITY_MASK
        elif index in subjects_set:
            return EntityMasks.SUBJ_ENTITY_MASK
        return EntityMasks.ANY_ENTITY_MASK
