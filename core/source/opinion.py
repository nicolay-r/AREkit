# -*- coding: utf-8 -*-
import io
import core.env as env
from core.source.entity import Entity
from core.source.synonyms import SynonymsCollection


class OpinionCollection:
    """ Collection of sentiment opinions between entities
    """

    def __init__(self, opinions, synonyms):
        assert(type(opinions) == list or type(opinions) == type(None))
        assert(isinstance(synonyms, SynonymsCollection))
        self.opinions = [] if opinions is None else opinions
        self.synonyms = synonyms
        self.by_value_index = self._create_index_by_value()
        self.by_synonym_index = self._create_index_by_synonyms(synonyms)

    def _create_index_by_value(self):
        index = set()
        for o in self.opinions:

            if not o.has_synonym_for_left(self.synonyms):
                self.synonyms.add_synonym(o.value_left)

            if not o.has_synonym_for_right(self.synonyms):
                self.synonyms.add_synonym(o.value_right)

            added = self._add_key(o.create_value_id(), index, check=False)
            if not added:
                print "Opinion with the values '{}'->'{}' already existed.".format(
                    o.value_left.encode('utf-8'), o.value_right.encode('utf-8'))

        return index

    def _create_index_by_synonyms(self, synonyms):
        index = set()

        for o in self.opinions:
            added = self._add_key(o.create_synonym_id(synonyms), index, check=False)
            if not added:
                print "Synonym of '{}'->'{}' already existed.".format(
                    o.value_left.encode('utf-8'), o.value_right.encode('utf-8'))
        return index

    @staticmethod
    def from_file(filepath, synonyms_filepath):
        synonyms = SynonymsCollection.from_file(synonyms_filepath)
        opinions = []
        with io.open(filepath, "r", encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):

                if line == '\n':
                    continue

                args = line.strip().split(',')

                if len(args) != 4:
                    print "should be 4 args at line: {}, '{}'".format(
                        i, line.encode('utf-8'))
                    continue

                entity_left = args[0].strip()
                entity_right = args[1].strip()

                o = Opinion(entity_left, entity_right, args[2].strip())
                opinions.append(o)

        return OpinionCollection(opinions, synonyms)

    def has_opinion_by_values(self, o):
        assert(isinstance(o, Opinion))
        return o.create_value_id() in self.by_value_index

    def has_opinion_by_synonyms(self, o):
        assert(isinstance(o, Opinion))

        if not o.has_synonym_for_left(self.synonyms):
            return False
        if not o.has_synonym_for_right(self.synonyms):
            return False

        return o.create_synonym_id(self.synonyms) in self.by_synonym_index

    def add_opinion(self, o):
        assert(isinstance(o, Opinion))

        if not o.has_synonym_for_left(self.synonyms):
            self.synonyms.add_synonym(o.value_left)

        if not o.has_synonym_for_right(self.synonyms):
            self.synonyms.add_synonym(o.value_right)

        self._add_key(o.create_value_id(), self.by_value_index)
        self._add_key(o.create_synonym_id(self.synonyms), self.by_synonym_index)
        self.opinions.append(o)

    def remove_opinion(self, o):
        assert(isinstance(o, Opinion))
        self.by_value_index.remove(o.create_value_id())
        self.by_synonym_index.remove(o.create_synonym_id(self.synonyms))
        self.opinions.remove(o)

    def limit(self, count):
        self.opinions = self.opinions[:count]

    def save(self, filepath):
        sorted_ops = sorted(self.opinions, key=lambda o: o.value_left + o.value_right)
        with io.open(filepath, 'w') as f:
            for o in sorted_ops:
                f.write(o.to_unicode())
                f.write(u'\n')

    @staticmethod
    def _add_key(key, collection, check=True):
        assert(type(key) == unicode)
        if check:
            assert(key not in collection)
        if key in collection:
            return False
        collection.add(key)
        return True

    def __len__(self):
        return len(self.opinions)

    def __iter__(self):
        for o in self.opinions:
            yield o


class Opinion:
    """ Source opinion description
    """

    def __init__(self, value_left, value_right, sentiment):
        assert(type(value_left) == unicode)
        assert(type(value_right) == unicode)
        assert(type(sentiment) == unicode)
        assert(',' not in value_left)
        assert(',' not in value_right)
        self.value_left = value_left.lower()
        self.value_right = value_right.lower()
        self.sentiment = sentiment

    @staticmethod
    def from_entities(entity_left, entity_right, sentiment):
        assert(isinstance(entity_left, Entity))
        assert(isinstance(entity_right, Entity))
        assert(type(sentiment) == unicode)
        return Opinion(entity_left.value, entity_right.value, sentiment)

    def to_unicode(self):
        return u"{}, {}, {}, current".format(
            self.value_left, self.value_right, self.sentiment)

    def create_value_id(self):
        return u"{}_{}".format(
            env.stemmer.lemmatize_to_str(self.value_left),
            env.stemmer.lemmatize_to_str(self.value_right))

    def create_synonym_id(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))
        return u"{}_{}".format(
            synonyms.get_synonym_group_index(self.value_left),
            synonyms.get_synonym_group_index(self.value_right))

    def has_synonym_for_left(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))
        return synonyms.has_synonym(self.value_left)

    def has_synonym_for_right(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))
        return synonyms.has_synonym(self.value_right)
