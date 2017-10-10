# -*- coding: utf-8 -*-
import io


class EntityCollection:
    """ Collection of annotated entities
    """

    def __init__(self, annots):
        self.annots = annots

    @staticmethod
    def from_file(filepath):
        """ Read annotation collection from file
        """

        annots = []
        with io.open(filepath, "r", encoding='utf-8') as f:
            for line in f.readlines():
                args = line.split()
                a = Entity(args[0], args[1], int(args[2]), int(args[3]),
                           args[4])
                annots.append(a)

        return EntityCollection(annots)

    def get(self, index):
        return self.annots[index]

    def count(self):
        return len(self.annots)

    def __iter__(self):
        for a in self.annots:
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

    def show(self):
        """ Displays annotation information
        """
        print "{}, {}, {}-{}, {}".format(
            self.ID.encode('utf-8'),
            self.str_type.encode('utf-8'),
            self.begin,
            self.end,
            self.value.encode('utf-8'))
