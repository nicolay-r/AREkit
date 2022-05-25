from arekit.common.entities.collection import EntityCollection
from arekit.common.synonyms import SynonymsCollection
from arekit.contrib.source.brat.entities.reader import BratEntityCollectionHelper
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions, RuSentRelIOUtils


class RuSentRelDocumentEntityCollection(EntityCollection):

    def __init__(self, entities, value_to_group_id_func):
        super(RuSentRelDocumentEntityCollection, self).__init__(
            entities=entities,
            value_to_group_id_func=value_to_group_id_func)

        self._sort_entities(key=lambda entity: entity.CharIndexBegin)

    @classmethod
    def read_collection(cls, doc_id, synonyms, version=RuSentRelVersions.V11):
        assert (isinstance(synonyms, SynonymsCollection))
        assert (isinstance(doc_id, int))

        return RuSentRelIOUtils.read_from_zip(
            inner_path=RuSentRelIOUtils.get_entity_innerpath(doc_id),
            process_func=lambda input_file: cls(
                entities=BratEntityCollectionHelper.extract_entities(input_file),
                value_to_group_id_func=synonyms.get_synonym_group_index),
            version=version)
