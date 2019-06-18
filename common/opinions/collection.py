# -*- coding: utf-8 -*-
from core.evaluation.labels import Label
from core.common.opinions.opinion import Opinion
from core.common.synonyms import SynonymsCollection


class OpinionCollection(object):
    """ Collection of sentiment opinions between entities
    """

    def __init__(self, opinions, synonyms):
        assert(isinstance(opinions, list) or isinstance(opinions, type(None)))
        assert(isinstance(synonyms, SynonymsCollection))
        self.__opinions = [] if opinions is None else opinions
        self.__synonyms = synonyms
        self.__by_synonyms = self.__create_index()

    def __add_synonym(self, value):
        if self.__synonyms.IsReadOnly:
            raise Exception((u"Failed to add '{}'. Synonym collection is read only!".format(value)).encode('utf-8'))
        self.__synonyms.add_synonym(value)

    def __create_index(self):
        index = {}
        for opinion in self.__opinions:
            OpinionCollection.__add_opinion(opinion, index, self.__synonyms, check=True)
        return index

    def has_synonymous_opinion(self, opinion, sentiment=None):
        assert(isinstance(opinion, Opinion))
        assert(sentiment is None or isinstance(sentiment, Label))

        if not opinion.has_synonym_for_source(self.__synonyms):
            return False
        if not opinion.has_synonym_for_target(self.__synonyms):
            return False

        s_id = opinion.create_synonym_id(self.__synonyms)
        if s_id in self.__by_synonyms:
            f_o = self.__by_synonyms[s_id]
            return True if sentiment is None else f_o.sentiment == sentiment

        return False

    def get_synonymous_opinion(self, opinion):
        assert(isinstance(opinion, Opinion))
        s_id = opinion.create_synonym_id(self.__synonyms)
        return self.__by_synonyms[s_id]

    def add_opinion(self, opinion):
        assert(isinstance(opinion, Opinion))

        if not opinion.has_synonym_for_source(self.__synonyms):
            self.__add_synonym(opinion.SourceValue)

        if not opinion.has_synonym_for_target(self.__synonyms):
            self.__add_synonym(opinion.TargetValue)

        self.__add_opinion(opinion, self.__by_synonyms, self.__synonyms)
        self.__opinions.append(opinion)

    @staticmethod
    def __add_opinion(opinion, collection, synonyms, check=True):
        key = opinion.create_synonym_id(synonyms)

        assert(isinstance(key, unicode))
        if check:
            if key in collection:
                raise Exception(u"'{}->{}' already exists in collection".format(
                    opinion.value_left, opinion.value_right).encode('utf-8'))
        if key in collection:
            return False
        collection[key] = opinion
        return True

    def iter_sentiment(self, sentiment):
        assert(isinstance(sentiment, Label))
        for o in self.__opinions:
            if o.sentiment == sentiment:
                yield o

    def __len__(self):
        return len(self.__opinions)

    def __iter__(self):
        for o in self.__opinions:
            yield o

