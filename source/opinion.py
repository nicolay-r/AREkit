# -*- coding: utf-8 -*-
import io

from core.processing.lemmatization.base import Stemmer
from core.evaluation.labels import Label
from core.source.synonyms import SynonymsCollection


class OpinionCollection:
    """ Collection of sentiment opinions between entities
    """

    def __init__(self, opinions, synonyms, stemmer, debug_mode=False):
        assert(isinstance(opinions, list) or isinstance(opinions, type(None)))
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(debug_mode, bool))
        self.debug_mode = debug_mode
        self.opinions = [] if opinions is None else opinions
        self.synonyms = synonyms
        self.stemmer = stemmer
        self.by_value_set = self._create_set_by_value()
        self.by_synonym_dict = self._create_dict_by_synonyms(synonyms)

    def __add_synonym(self, value):
        if self.synonyms.IsReadOnly:
            raise Exception((u"Failed to add '{}'. Synonym collection is read only!".format(value)).encode('utf-8'))
        self.synonyms.add_synonym(value)

    def _create_set_by_value(self):
        index = set()
        for o in self.opinions:

            if not o.has_synonym_for_left(self.synonyms):
                self.__add_synonym(o.value_left)

            if not o.has_synonym_for_right(self.synonyms):
                self.__add_synonym(o.value_right)

            added = self._add_key(o.create_value_id(), index, check=False)
            if not added and self.debug_mode:
                print "Opinion with the values '{}'->'{}' already existed.".format(
                    o.value_left.encode('utf-8'), o.value_right.encode('utf-8'))

        return index

    def _create_dict_by_synonyms(self, synonyms):
        index = {}
        for o in self.opinions:
            added = self._add_key_value(o.create_synonym_id(synonyms), o, index, check=False)
            if not added and self.debug_mode:
                print "Synonym of '{}'->'{}' already existed.".format(
                    o.value_left.encode('utf-8'), o.value_right.encode('utf-8'))
        return index

    @classmethod
    def from_file(cls, filepath, synonyms, stemmer, debug=False):
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(stemmer, Stemmer))

        opinions = []
        filepaths = []
        if isinstance(filepath, unicode):
            filepaths.append(filepath)
        elif isinstance(filepath, list):
            filepaths = filepath

        for fp in filepaths:
            with io.open(fp, "r", encoding='utf-8') as f:
                for i, line in enumerate(f.readlines()):

                    if line == '\n':
                        continue

                    args = line.strip().split(',')

                    if len(args) < 3:
                        print "should be at least 3 arguments: {}, '{}'".format(
                            i, line.encode('utf-8'))
                        continue

                    entity_left = args[0].strip()
                    entity_right = args[1].strip()
                    sentiment = Label.from_str(args[2].strip())

                    o = Opinion(entity_left, entity_right, sentiment, stemmer)
                    opinions.append(o)

        return cls(opinions, synonyms, stemmer, debug)

    def has_opinion_by_values(self, o):
        assert(isinstance(o, Opinion))
        return o.create_value_id() in self.by_value_set

    def has_opinion_by_synonyms(self, o, sentiment=None):
        assert(isinstance(o, Opinion))
        assert(sentiment is None or isinstance(sentiment, Label))

        if not o.has_synonym_for_left(self.synonyms):
            return False
        if not o.has_synonym_for_right(self.synonyms):
            return False

        s_id = o.create_synonym_id(self.synonyms)
        if s_id in self.by_synonym_dict:
            f_o = self.by_synonym_dict[s_id]
            return True if sentiment is None else f_o.sentiment == sentiment

        return False

    def get_opinion_by_synonyms(self, o):
        assert(isinstance(o, Opinion))
        s_id = o.create_synonym_id(self.synonyms)
        return self.by_synonym_dict[s_id]

    def add_opinion(self, o):
        assert(isinstance(o, Opinion))
        # stemmers should be the same
        assert(o.stemmer is self.stemmer)

        if not o.has_synonym_for_left(self.synonyms):
            self.__add_synonym(o.value_left)

        if not o.has_synonym_for_right(self.synonyms):
            self.__add_synonym(o.value_right)

        self._add_key(o.create_value_id(), self.by_value_set)
        self._add_key_value(o.create_synonym_id(self.synonyms), o, self.by_synonym_dict)
        self.opinions.append(o)

    def remove_opinion(self, o):
        assert(isinstance(o, Opinion))
        self.by_value_set.remove(o.create_value_id())
        del self.by_synonym_dict[o.create_synonym_id(self.synonyms)]
        self.opinions.remove(o)

    def create_opinion(self, value_left, value_right, sentiment):
        """
        Create opinion with the same stemmer as in collection
        """
        return Opinion(value_left=value_left,
                       value_right=value_right,
                       sentiment=sentiment,
                       stemmer=self.stemmer)

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
        assert(isinstance(key, unicode))
        if check:
            assert(key not in collection)
        if key in collection:
            return False
        collection.add(key)
        return True

    @staticmethod
    def _add_key_value(key, value, collection, check=True):
        assert(isinstance(key, unicode))
        if check:
            assert(key not in collection)
        if key in collection:
            return False
        collection[key] = value
        return True

    def iter_sentiment(self, sentiment):
        assert(isinstance(sentiment, Label))
        for o in self.opinions:
            if o.sentiment == sentiment:
                yield o

    def __len__(self):
        return len(self.opinions)

    def __iter__(self):
        for o in self.opinions:
            yield o


class Opinion:
    """ Source opinion description
    """

    # TODO. Use create opinion from collection.
    def __init__(self, value_left, value_right, sentiment, stemmer):
        assert(isinstance(value_left, unicode))
        assert(isinstance(value_right, unicode))
        assert(isinstance(sentiment, Label))
        assert(isinstance(stemmer, Stemmer))
        assert(',' not in value_left)
        assert(',' not in value_right)
        self.value_left = value_left.lower()
        self.value_right = value_right.lower()
        self.sentiment = sentiment
        self.stemmer = stemmer

    # TODO: Should be a part of collection during save operation.
    def to_unicode(self):
        return u"{}, {}, {}, current".format(
            self.value_left,
            self.value_right,
            self.sentiment.to_str())

    def create_value_id(self):
        return u"{}_{}".format(
            self.stemmer.lemmatize_to_str(self.value_left),
            self.stemmer.lemmatize_to_str(self.value_right))

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
