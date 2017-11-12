# -*- coding: utf-8 -*-
import io
import core.env as env


class EntityCollection:
    """ Collection of annotated entities
    """

    def __init__(self, entities):
        self.entities = entities
        self.by_id = self._index_by_id()
        self.by_lemmas = self._index_by_lemmas()

    def _index_by_id(self):
        index = {}
        for e in self.entities:
            index[e.ID] = e
        return index

    def _index_by_lemmas(self):
        index = {}
        for e in self.entities:
            key = self._create_lemma(e.value)
            if key in index:
                index[key].append(e.ID)
            else:
                index[key] = [e.ID]
        return index

    @staticmethod
    def from_file(filepath):
        """ Read annotation collection from file
        """
        entities = []
        with io.open(filepath, "r", encoding='utf-8') as f:
            for line in f.readlines():
                args = line.split()

                e_ID = args[0]
                e_str_type = args[1]
                e_begin = int(args[2])
                e_end = int(args[3])
                e_value = " ".join([a.strip() for a in args[4:]])
                a = Entity(e_ID, e_str_type, e_begin, e_end, e_value)

                entities.append(a)

        # sort by beginning
        entities.sort(key=lambda e: e.begin)

        return EntityCollection(entities)

    def has_entity_by_value(self, entity_value):
        assert(type(entity_value) == unicode)
        lemma = self._create_lemma(entity_value)
        return lemma in self.by_lemmas

    def get_entity_by_index(self, index):
        return self.entities[index]

    def get_entity_by_id(self, ID):
        assert(type(ID) == unicode)
        return self.by_id[ID]

    def get_entity_by_value(self, entity_value):
        assert(type(entity_value) == unicode)
        lemma = self._create_lemma(entity_value)
        return self.by_lemmas[lemma]

    def _create_lemma(self, entity_value):
        assert(type(entity_value) == unicode)
        return env.stemmer.lemmatize_to_str(entity_value)

    def count(self):
        return len(self.entities)

    def __iter__(self):
        for a in self.entities:
            yield a


class Entity:
    """ Entity description.
    """

    def __init__(self, ID, str_type, begin, end, value):
        assert(type(ID) == unicode)
        assert(type(str_type) == unicode)
        assert(type(begin) == int)
        assert(type(end) == int)
        assert(type(value) == unicode and len(value) > 0)
        self.ID = ID
        self.str_type = str_type
        self.begin = begin
        self.end = end
        self.value = value.lower()

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
