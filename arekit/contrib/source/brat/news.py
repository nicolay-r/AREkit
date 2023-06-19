from arekit.common.docs.base import Document
from arekit.contrib.source.brat.entities.entity import BratEntity
from arekit.contrib.source.brat.sentence import BratSentence


class BratDocument(Document):

    def __init__(self, doc_id, sentences, text_relations):
        assert(isinstance(text_relations, list) or text_relations is None)
        super(BratDocument, self).__init__(doc_id=doc_id, sentences=sentences)
        self.__text_relations = text_relations
        self.__entity_by_id = {}
        for sentence in sentences:
            assert(isinstance(sentence, BratSentence))
            for brat_entity, _ in sentence.iter_entity_with_local_bounds():
                assert(isinstance(brat_entity, BratEntity))
                self.__entity_by_id[brat_entity.ID] = brat_entity

    @property
    def Relations(self):
        for brat_relation in self.__text_relations:
            yield brat_relation

    def contains_entity(self, entity_id):
        return entity_id in self.__entity_by_id

    def get_entity_by_id(self, entity_id):
        return self.__entity_by_id[entity_id]
