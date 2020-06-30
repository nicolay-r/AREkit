from enum import Enum

from arekit.common.news.parsed.term_position import TermPositionTypes


class EntityEndType(Enum):
    Source = 1
    Target = 2


class DistanceType(Enum):
    InTerms = 1
    InSentences = 2

    @staticmethod
    def to_position_type(dist_type):
        assert(isinstance(dist_type, DistanceType))

        if dist_type == DistanceType.InTerms:
            return TermPositionTypes.IndexInDocument

        if dist_type == DistanceType.InSentences:
            return TermPositionTypes.SentenceIndex
