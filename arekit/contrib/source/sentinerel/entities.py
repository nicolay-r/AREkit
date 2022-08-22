from arekit.common.entities.collection import EntityCollection
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.contrib.source.brat.annot import BratAnnotationParser
from arekit.contrib.source.brat.entities.entity import BratEntity
from arekit.contrib.source.sentinerel.io_utils import SentiNerelIOUtils
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection


class SentiNerelEntityCollection(EntityCollection):

    entities_to_ignore = [
        "EFFECT_NEG",
        "EFFECT_POS",
        "ARGUMENT_NEG",
        "ARGUMENT_POS",
        "EVENT"
    ]

    def __init__(self, contents, value_to_group_id_func):
        assert(isinstance(contents, dict))
        assert("entities" in contents)

        self.__dicard_entities = set(self.entities_to_ignore)

        contents["entities"] = [e for e in contents["entities"] if self.__keep_entity(e)]

        super(SentiNerelEntityCollection, self).__init__(
            entities=contents["entities"],
            value_to_group_id_func=value_to_group_id_func)

        self._sort_entities(key=lambda entity: entity.IndexBegin)

    def __keep_entity(self, entity):
        assert(isinstance(entity, BratEntity))
        return entity.Type not in self.__dicard_entities

    @classmethod
    def read_collection(cls, filename, version):
        assert(isinstance(filename, str))

        # Since this dataset does not provide the synonyms collection by default,
        # it is necessary to declare an empty collection to populate so in further.
        synonyms = StemmerBasedSynonymCollection(iter_group_values_lists=[],
                                                 stemmer=MystemWrapper(),
                                                 is_read_only=False,
                                                 debug=False)

        return SentiNerelIOUtils.read_from_zip(
            inner_path=SentiNerelIOUtils.get_annotation_innerpath(filename),
            process_func=lambda input_file: cls(
                contents=BratAnnotationParser.parse_annotations(input_file=input_file, encoding='utf-8-sig'),
                value_to_group_id_func=lambda value:
                    SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                        synonyms, value)),
            version=version)
