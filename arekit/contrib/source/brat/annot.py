from arekit.contrib.source.brat.entities.entity import BratEntity
from arekit.contrib.source.brat.relation import BratRelation


class BratAnnotationParser:

    ENTITIES = "entities"
    RELATIONS = "relations"

    @staticmethod
    def __non_prefixed_id(value):
        assert (isinstance(value, str))
        return value[1:]

    @staticmethod
    def handle_entity(args):
        """ T2	Location 10 23	South America
            T1	Location 0 5;16 23	North America
        """
        assert(len(args) == 3)

        e_id = int(BratAnnotationParser.__non_prefixed_id(args[0]))
        entity_params = args[1].split()

        if len(entity_params) != 3:
            # We do not support the case of a non-continuous entity mentions.
            return None

        e_str_type, e_begin, e_end = entity_params

        return BratEntity(id_in_doc=e_id,
                          e_type=e_str_type,
                          index_begin=int(e_begin),
                          index_end=int(e_end),
                          childs=None,
                          value=args[2].strip())

    @staticmethod
    def handle_relation(args):
        """ Example:
            R1	Origin Arg1:T3 Arg2:T4
        """

        # Parse identifier index.
        e_id = args[0][1:]

        # Parse relation arguments.
        rel_type, source, target = args[1].split()

        source_id = source.split(':')[1]
        target_id = target.split(':')[1]

        return BratRelation(id_in_doc=e_id,
                            source_id=int(BratAnnotationParser.__non_prefixed_id(source_id)),
                            target_id=int(BratAnnotationParser.__non_prefixed_id(target_id)),
                            rel_type=rel_type)

    @staticmethod
    def parse_annotations(input_file, encoding='utf-8'):
        """ Read annotation collection from file
        """
        entities = []
        relations = []

        for line in input_file.readlines():
            line = line.decode(encoding)

            args = line.split('\t')

            record_type = args[0][0]

            # Entities (objects) are prefixed with `T`
            if record_type == "T":
                entity = BratAnnotationParser.handle_entity(args)
                if entity is not None:
                    entities.append(entity)

            elif record_type == "R":
                relations.append(BratAnnotationParser.handle_relation(args))

        return {
            BratAnnotationParser.ENTITIES: entities,
            BratAnnotationParser.RELATIONS: relations
        }
