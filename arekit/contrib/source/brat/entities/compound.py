from arekit.contrib.source.brat.entities.entity import BratEntity


class BratCompoundEntity(BratEntity):
    """ Entity which contains the hierarchy of the other entities.
    """

    def __init__(self, id_in_doc, value, e_type, root, entities, index_begin, index_end, group_index=None):
        assert(isinstance(entities, list))
        assert(isinstance(root, BratCompoundEntity) or root is None)
        super(BratCompoundEntity, self).__init__(value=value, e_type=e_type,
                                                 id_in_doc=id_in_doc,
                                                 index_begin=index_begin,
                                                 index_end=index_end,
                                                 group_index=group_index)
        self.__entities = entities
        self.__root = root

    @property
    def Root(self):
        return self.__root

    def iter_childs(self):
        return iter(self.__entities)
