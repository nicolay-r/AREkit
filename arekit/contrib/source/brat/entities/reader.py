# -*- coding: utf-8 -*-
from arekit.contrib.source.brat.entities.entity import BratEntity


class BratEntityCollectionHelper:

    @staticmethod
    def extract_entities(input_file, encoding='utf-8'):
        """ Read annotation collection from file
        """
        entities = []

        for line in input_file.readlines():
            line = line.decode(encoding)

            args = line.split()

            record_type = args[0][1]
            # Entities (objects) are prefixed with `T`
            if record_type != 'T':
                continue

            e_id = int(args[0][1:])
            e_str_type = args[1]
            e_begin = int(args[2])
            e_end = int(args[3])
            e_value = " ".join([arg.strip().replace(',', '') for arg in args[4:]])

            entity = BratEntity(id_in_doc=e_id,
                                e_type=e_str_type,
                                char_index_begin=e_begin,
                                char_index_end=e_end,
                                value=e_value)

            entities.append(entity)

        return entities
