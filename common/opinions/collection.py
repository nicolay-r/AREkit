# -*- coding: utf-8 -*-
import collections

from arekit.common.labels.base import Label
from arekit.common.opinions.base import Opinion
from arekit.common.synonyms import SynonymsCollection


class OpinionCollection(object):
    """
    Document-level Collection of sentiment opinions between entities
    """

    def __init__(self, opinions, synonyms, raise_exception_on_duplicates):
        """
        opinions:
        synonyms:
        raise_exception_on_duplicates: bool
            denotes whether there is a need to fire exception for duplicates in opinions list.
        """
        assert(isinstance(opinions, collections.Iterable) or isinstance(opinions, type(None)))
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(raise_exception_on_duplicates, bool))

        self.__by_synonyms = {}
        self.__ordered_opinion_keys = []
        self.__synonyms = synonyms

        if opinions is None:
            return

        for opinion in opinions:
            self.__register_opinion(opinion=opinion,
                                    raise_exception_on_existence=raise_exception_on_duplicates)

    # region class methods

    @classmethod
    def init_as_custom(cls, opinions, synonyms):
        return cls(opinions=opinions,
                   synonyms=synonyms,
                   raise_exception_on_duplicates=False)

    @classmethod
    def create_empty(cls, synonyms):
        return cls(opinions=[],
                   synonyms=synonyms,
                   raise_exception_on_duplicates=True)

    # endregion

    # region public methods

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

        self.__register_opinion(opinion=opinion,
                                raise_exception_on_existence=True)

    def iter_sentiment(self, sentiment):
        assert(isinstance(sentiment, Label))
        for key in self.__ordered_opinion_keys:
            opinion = self.__by_synonyms[key]
            if opinion.sentiment == sentiment:
                yield opinion

    # endregion

    # region private methods

    def __add_synonym(self, value):
        self.__synonyms.add_synonym_value(value)

    def __register_opinion(self, opinion, raise_exception_on_existence):
        key = opinion.create_synonym_id(self.__synonyms)

        assert(isinstance(key, unicode))
        if key in self.__by_synonyms:

            if raise_exception_on_existence:
                raise Exception(u"'{}->{}' already exists in collection".format(
                    opinion.SourceValue, opinion.TargetValue).encode('utf-8'))

            # Rejecting opinion.
            return False

        # Perform registration.
        self.__by_synonyms[key] = opinion
        self.__ordered_opinion_keys.append(key)

        return True

    # endregion

    # region base methods

    def __len__(self):
        return len(self.__by_synonyms)

    def __iter__(self):
        for key in self.__ordered_opinion_keys:
            yield self.__by_synonyms[key]

    # endregion
