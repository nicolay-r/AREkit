from arekit.common.entities.base import Entity


class DocumentEntity(Entity):

    def __init__(self, value, e_type, id_in_doc, group_index):
        super(DocumentEntity, self).__init__(value=value,
                                             e_type=e_type,
                                             group_index=group_index)
        self.__id = id_in_doc

    @property
    def IdInDocument(self):
        return self.__id
