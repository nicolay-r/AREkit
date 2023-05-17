from arekit.common.entities.collection import EntityCollection
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.contrib.source.brat.annot import BratAnnotationParser
from arekit.contrib.source.brat.entities.entity import BratEntity
from arekit.contrib.source.sentinerel.io_utils import SentiNerelIOUtils
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection


class SentiNerelEntityCollection(EntityCollection):

    def __init__(self, contents, value_to_group_id_func, entities_to_ignore=None):
        """
            entities_to_ignore: list or None
                this parameter is required because of the simplified implmentation of
                the nested objects of the BRAT annotation.
        """
        assert(isinstance(contents, dict))
        assert(BratAnnotationParser.ENTITIES in contents)
        assert(isinstance(entities_to_ignore, list) or entities_to_ignore is None)

        self.__dicard_entities = set([] if entities_to_ignore is None else entities_to_ignore)
        contents[BratAnnotationParser.ENTITIES] = [e for e in contents[BratAnnotationParser.ENTITIES]
                                                   if self.__keep_entity(e)]

        super(SentiNerelEntityCollection, self).__init__(
            entities=contents[BratAnnotationParser.ENTITIES],
            value_to_group_id_func=value_to_group_id_func)

        self._sort_entities(key=lambda entity: entity.IndexBegin)

    def __keep_entity(self, entity):
        assert(isinstance(entity, BratEntity))
        return entity.Type not in self.__dicard_entities

    @classmethod
    def read_collection(cls, filename, version, entities_to_ignore=None):
        assert(isinstance(filename, str))

        # Since this dataset does not provide the synonyms collection by default,
        # it is necessary to declare an empty collection to populate so in further.
        synonyms = StemmerBasedSynonymCollection(stemmer=MystemWrapper(), is_read_only=False)

        return SentiNerelIOUtils.read_from_zip(
            inner_path=SentiNerelIOUtils.get_annotation_innerpath(filename),
            process_func=lambda input_file: cls(
                contents=BratAnnotationParser.parse_annotations(input_file=input_file, encoding='utf-8-sig'),
                entities_to_ignore=entities_to_ignore,
                value_to_group_id_func=lambda value:
                    SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                        synonyms, value)),
            version=version)
