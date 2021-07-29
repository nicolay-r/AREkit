from arekit.common.entities.base import Entity
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.term_position import TermPositionTypes, TermPosition
from arekit.common.text_opinions.base import TextOpinion
from arekit.common.dataset.text_opinions.enums import EntityEndType, DistanceType


class TextOpinionHelper(object):
    """
    This class provides a helper functions for TextOpinions, which become a part of TextOpinionCollection.
    The latter is important because of the dependency from Owner.
    We utilize 'extract' prefix in methods to emphasize that these are methods of helper.

    Wrapper over:
        parsed news, positions, text_opinions
    """

    # region public 'extract' methods

    @staticmethod
    def extract_entity_value(parsed_news, text_opinion, end_type):
        return TextOpinionHelper.__extract_entity_value(parsed_news=parsed_news,
                                                        text_opinion=text_opinion,
                                                        end_type=end_type)

    @staticmethod
    def extract_entity_position(parsed_news, text_opinion, end_type, position_type=None):
        return TextOpinionHelper.__get_entity_position(parsed_news=parsed_news,
                                                       text_opinion=text_opinion,
                                                       end_type=end_type,
                                                       position_type=position_type)

    # endregion

    # region public 'calculate' methods

    @staticmethod
    def calc_dist_between_text_opinion_ends(parsed_news, text_opinion, distance_type):
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(distance_type, DistanceType))
        assert(isinstance(parsed_news, ParsedNews))
        assert(parsed_news.RelatedNewsID == text_opinion.NewsID)

        e1_id = TextOpinionHelper.__get_end_id(text_opinion=text_opinion,
                                               end_type=EntityEndType.Source)

        e2_id = TextOpinionHelper.__get_end_id(text_opinion=text_opinion,
                                               end_type=EntityEndType.Target)

        return TextOpinionHelper.__calc_distance(
            pos1=parsed_news.get_entity_position(id_in_document=e1_id),
            pos2=parsed_news.get_entity_position(id_in_document=e2_id),
            position_type=DistanceType.to_position_type(distance_type))

    @staticmethod
    def calc_dist_between_text_opinion_end_indices(pos1_ind, pos2_ind):
        return TextOpinionHelper.__calc_distance_by_inds(pos1_ind=pos1_ind, pos2_ind=pos2_ind)

    @staticmethod
    def calc_dist_between_entities(parsed_news, e1, e2, distance_type):
        assert(isinstance(e1, Entity))
        assert(isinstance(e2, Entity))
        assert(isinstance(distance_type, DistanceType))

        return TextOpinionHelper.__calc_distance(
            pos1=parsed_news.get_entity_position(e1.IdInDocument),
            pos2=parsed_news.get_entity_position(e2.IdInDocument),
            position_type=DistanceType.to_position_type(distance_type))

    # endregion

    # region private methods

    @staticmethod
    def __extract_entity_value(parsed_news, text_opinion, end_type):
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(text_opinion, TextOpinion))
        assert(parsed_news.RelatedNewsID == text_opinion.NewsID)
        end_id = TextOpinionHelper.__get_end_id(text_opinion=text_opinion,
                                                end_type=end_type)
        return parsed_news.get_entity_value(end_id)

    @staticmethod
    def __get_entity_position(parsed_news, text_opinion, end_type, position_type=None):
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(text_opinion, TextOpinion))
        assert(parsed_news.RelatedNewsID == text_opinion.NewsID)
        end_id = TextOpinionHelper.__get_end_id(text_opinion=text_opinion,
                                                             end_type=end_type)
        return parsed_news.get_entity_position(end_id, position_type)

    @staticmethod
    def __calc_distance(pos1, pos2, position_type=TermPositionTypes.IndexInDocument):
        assert(isinstance(pos1, TermPosition))
        assert(isinstance(pos2, TermPosition))
        return TextOpinionHelper.__calc_distance_by_inds(pos1_ind=pos1.get_index(position_type),
                                                         pos2_ind=pos2.get_index(position_type))

    @staticmethod
    def __calc_distance_by_inds(pos1_ind, pos2_ind):
        return abs(pos1_ind - pos2_ind)

    @staticmethod
    def __get_end_id(text_opinion, end_type):
        assert(isinstance(text_opinion, TextOpinion))
        assert(end_type == EntityEndType.Source or end_type == EntityEndType.Target)
        return text_opinion.SourceId if end_type == EntityEndType.Source else text_opinion.TargetId

    # endregion
