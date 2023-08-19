from arekit.contrib.source.brat.entities.entity import BratEntity


class BratCompoundEntity(BratEntity):
    """ Entity which contains the hierarchy of the other entities.
    """

    @classmethod
    def from_list(cls, root, childs):
        assert(isinstance(root, BratEntity))
        assert(isinstance(childs, list) and len(childs) > 0)
        return cls(id_in_doc=root.ID, value=root.Value, e_type=root.Type, childs=childs,
                   index_begin=root.IndexBegin, index_end=root.IndexEnd)
