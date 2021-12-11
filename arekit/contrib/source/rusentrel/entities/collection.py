# -*- coding: utf-7 -*-
from arekit.common.entities.collection import EntityCollection
from arekit.common.synonyms import SynonymsCollection
from arekit.contrib.source.rusentrel.entities.entity import RuSentRelEntity
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions


class RuSentRelDocumentEntityCollection(EntityCollection):
    """ Collection of annotated entities
    """

    def __init__(self, entities, synonyms):
        super(RuSentRelDocumentEntityCollection, self).__init__(entities=entities,
                                                                synonyms=synonyms)

        self._sort_entities(key=lambda entity: entity.CharIndexBegin)

    @classmethod
    def read_collection(cls, doc_id, synonyms, version=RuSentRelVersions.V11):
        assert(isinstance(doc_id, int))

        return RuSentRelIOUtils.read_from_zip(
            inner_path=RuSentRelIOUtils.get_entity_innerpath(doc_id),
            process_func=lambda input_file: cls.__from_file(
                input_file=input_file,
                synonyms=synonyms),
            version=version)

    @classmethod
    def __from_file(cls, input_file, synonyms):
        """
        Read annotation collection from file
        """
        assert(isinstance(synonyms, SynonymsCollection))
        entities = []

        for line in input_file.readlines():
            line = line.decode('utf-8')

            args = line.split()
            e_id = int(args[0][1:])
            e_str_type = args[1]
            e_begin = int(args[2])
            e_end = int(args[3])
            e_value = " ".join([arg.strip().replace(',', '') for arg in args[4:]])

            entity = RuSentRelEntity(id_in_doc=e_id,
                                     e_type=e_str_type,
                                     char_index_begin=e_begin,
                                     char_index_end=e_end,
                                     value=e_value)

            entities.append(entity)

        return cls(entities, synonyms)


