import collections
import logging

from arekit.common.labels.base import Label
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.enums import OpinionEndTypes
from arekit.common.synonyms import SynonymsCollection


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class OpinionCollection(object):
    """
    Document-level Collection of sentiment opinions between entities
    """

    def __init__(self, opinions, synonyms,
                 error_on_duplicates,
                 error_on_synonym_end_missed):
        """
        opinions:
        synonyms:
        raise_exception_on_duplicates: bool
            denotes whether there is a need to fire exception for duplicates in opinions list.
        """
        assert(isinstance(opinions, collections.Iterable) or isinstance(opinions, type(None)))
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(error_on_duplicates, bool))
        assert(isinstance(error_on_synonym_end_missed, bool))

        self.__by_synonyms = {}
        self.__ordered_opinion_keys = []
        self.__synonyms = synonyms

        if opinions is None:
            return

        for opinion in opinions:
            self.__register_opinion(
                opinion=opinion,
                error_on_existence=error_on_duplicates,
                error_on_synonym_end_missed=error_on_synonym_end_missed)

    # region class methods

    @classmethod
    def init_as_custom(cls, opinions, synonyms):
        """
        Perform initialization with a synonyms collection which might be partially incompatible
        with opinion ends (values).
        """
        return cls(opinions=opinions,
                   synonyms=synonyms,
                   error_on_duplicates=False,
                   error_on_synonym_end_missed=False)

    @classmethod
    def create_empty(cls, synonyms):
        return cls(opinions=[],
                   synonyms=synonyms,
                   error_on_duplicates=True,
                   error_on_synonym_end_missed=True)

    # endregion

    # region public methods

    def has_synonymous_opinion(self, opinion, sentiment=None):
        assert(isinstance(opinion, Opinion))
        assert(sentiment is None or isinstance(sentiment, Label))

        for end_type in OpinionEndTypes:
            if not opinion.has_synonym_for_end(synonyms=self.__synonyms, end_type=end_type):
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

        self.__register_opinion(opinion=opinion,
                                error_on_existence=True,
                                error_on_synonym_end_missed=True)

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

    def __register_opinion(self, opinion,
                           error_on_existence,
                           error_on_synonym_end_missed):
        assert(isinstance(error_on_existence, bool))
        assert(isinstance(error_on_synonym_end_missed, bool))

        for end_type in OpinionEndTypes:
            value = opinion.get_value(end_type)
            if opinion.has_synonym_for_end(synonyms=self.__synonyms, end_type=end_type):
                # OK.
                continue
            if not self.__synonyms.IsReadOnly:
                # OK. Registering new synonyms as it is possible.
                self.__add_synonym(value)
                continue

            message = u"'{s}' for end {e} does not exist in read-only SynonymsCollection".format(
                s=value,
                e=end_type).encode('utf-8')
            if error_on_synonym_end_missed:
                raise Exception(message)
            else:
                # Rejecting opinion.
                logger.info(message)
                return False

        if opinion.is_loop(self.__synonyms):
            # Ignoring loops.
            return False

        key = opinion.create_synonym_id(self.__synonyms)

        assert(isinstance(key, unicode))
        if key in self.__by_synonyms:

            message = u"'{s}->{t}' already exists in collection".format(
                s=opinion.SourceValue,
                t=opinion.TargetValue).encode('utf-8')

            if error_on_existence:
                raise Exception(message)
            else:
                logger.info(message)
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
