from arekit.common.entities.base import Entity
from arekit.common.docs.entity import DocumentEntity
from arekit.common.docs.parsed.base import ParsedDocument


class BaseParsedDocumentServiceProvider(object):

    def __init__(self, entity_index_func=None):
        """ Outside enity indexing function
            entity_index_func: provides id for a given entity, i.e.
                func(entity) -> int (id)
        """
        assert(callable(entity_index_func) or entity_index_func is None)
        self._doc_entities = None
        self.__entity_map = {}
        self.__entity_index_func = entity_index_func

    @property
    def Name(self):
        raise NotImplementedError()

    def init_parsed_doc(self, parsed_doc):
        assert(isinstance(parsed_doc, ParsedDocument))

        def __iter_childs_and_root_node(entity):
            """ Note: Entity has childs and we would like to iterate over childs
                to conider them as well as keep the root Node.
            """
            # We first add childs.
            for child_entity in entity.iter_childs():
                yield child_entity, True

            # Return Root node.
            yield entity, False

        self._doc_entities = []
        self.__entity_map.clear()

        current_id = 0
        for _, entity in enumerate(parsed_doc.iter_entities()):

            child_doc_entities = []
            for tree_entity, is_child in __iter_childs_and_root_node(entity):

                doc_entity = DocumentEntity(id_in_doc=current_id,
                                            value=tree_entity.Value,
                                            e_type=tree_entity.Type,
                                            display_value=tree_entity.DisplayValue,
                                            childs=None if is_child else child_doc_entities,
                                            group_index=tree_entity.GroupIndex)
                current_id += 1

                if is_child:
                    child_doc_entities.append(doc_entity)

                self._doc_entities.append(doc_entity)

                if self.__entity_index_func is not None:
                    self.__entity_map[self.__entity_index_func(tree_entity)] = doc_entity

    def get_document_entity(self, entity):
        """ Maps entity to the related one with DocumentEntity type
        """
        assert(isinstance(entity, Entity))
        return self.__entity_map[self.__entity_index_func(entity)]

    def contains_entity(self, entity):
        return self.__entity_index_func(entity) in self.__entity_map
