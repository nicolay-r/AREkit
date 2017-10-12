# -*- coding: utf-8 -*-
import io


class EntityCollection:
    """ Collection of annotated entities
    """

    def __init__(self, entities):
        self.entities = entities

    @staticmethod
    def from_file(filepath):
        """ Read annotation collection from file
        """

        entities = []
        with io.open(filepath, "r", encoding='utf-8') as f:
            for line in f.readlines():
                args = line.split()
                a = Entity(args[0], args[1], int(args[2]), int(args[3]),
                           args[4])
                entities.append(a)

        # sort by beginning
        entities.sort(key=lambda e: e.begin)

        return EntityCollection(entities)

    def get(self, index):
        return self.entities[index]

    def count(self):
        return len(self.entities)

    def __iter__(self):
        for a in self.entities:
            yield a


class Entity:
    """ Entity description
    """

    def __init__(self, ID, str_type, begin, end, value):
        assert(type(ID) == unicode)
        assert(type(str_type) == unicode)
        assert(type(begin) == int)
        assert(type(end) == int)
        assert(type(value) == unicode)
        self.ID = ID
        self.str_type = str_type
        self.begin = begin
        self.end = end
        self.value = value

    def get_int_ID(self):
        return int(self.ID[1:len(self.ID)])

    def show(self):
        """ Displays annotation information
        """
        print "{}, {}, {}-{}, {}".format(
            self.ID.encode('utf-8'),
            self.str_type.encode('utf-8'),
            self.begin,
            self.end,
            self.value.encode('utf-8'))
