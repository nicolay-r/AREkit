from arekit.common.entities.base import Entity
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.parsed_news.term_position import TermPosition, TermPositionTypes
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.common.text_opinions.collection import TextOpinionCollection
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.common.text_opinions.end_type import EntityEndType


# TODO. MOVE as written below
# TODO. This should be moved into dataset dir
# TODO. As helper
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
    def extract_entity_value(text_opinion, end_type):
        parsed_news, id = TextOpinionHelper.__get(text_opinion, end_type)
        return parsed_news.get_entity_value(id)

    @staticmethod
    def extract_entity_sentence_index(text_opinion, end_type):
        entity_position = TextOpinionHelper.__get_entity_position(text_opinion, end_type)
        return entity_position.get_index(TermPositionTypes.SentenceIndex)

    @staticmethod
    def extract_entity_sentence_level_term_index(text_opinion, end_type):
        parsed_news, id = TextOpinionHelper.__get(text_opinion, end_type)
        term_position = parsed_news.get_entity_position(id)
        return term_position.get_index(TermPositionTypes.IndexInSentence)

    @staticmethod
    def extract_entity_sentence_level_synonym_indices(text_opinion, end_type, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))
        parsed_news, id = TextOpinionHelper.__get(text_opinion, end_type)

        entity_pos = parsed_news.get_entity_position(id)
        s_index = entity_pos.get_index(TermPositionTypes.SentenceIndex)

        e_value = parsed_news.get_entity_value(id)
        e_group = synonyms.get_synonym_group_index(e_value)

        inds = []

        it = parsed_news.iter_sentence_terms(
            sentence_index=s_index,
            term_check=lambda term: isinstance(term, Entity))

        for e_index, e in it:

            if not synonyms.contains_synonym_value(e.Value):
                if e_value != e.Value:
                    continue
            elif e_group != synonyms.get_synonym_group_index(e.Value):
                continue

            inds.append(e_index)

        return inds

    # endregion

    # region public 'calculate' methods

    @staticmethod
    def calculate_distance_between_text_opinion_ends_in_terms(text_opinion):
        assert(isinstance(text_opinion, TextOpinion))

        pos1 = TextOpinionHelper.__get_entity_position(text_opinion=text_opinion,
                                                       end_type=EntityEndType.Source)

        pos2 = TextOpinionHelper.__get_entity_position(text_opinion=text_opinion,
                                                       end_type=EntityEndType.Target)

        return TextOpinionHelper.__calc_distance_in_terms(pos1=pos1, pos2=pos2)

    @staticmethod
    def calculate_distance_between_entities_in_terms(parsed_news, e1, e2):
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(e1, Entity))
        assert(isinstance(e2, Entity))

        return TextOpinionHelper.__calc_distance_in_terms(
            pos1=parsed_news.get_entity_position(e1.IdInDocument),
            pos2=parsed_news.get_entity_position(e2.IdInDocument))

    # endregion

    # region public 'check' methods

    @staticmethod
    def check_ends_has_same_sentence_index(text_opinion):
        assert(isinstance(text_opinion, TextOpinion))

        pos1 = TextOpinionHelper.__get_entity_position(text_opinion, EntityEndType.Source)
        pos2 = TextOpinionHelper.__get_entity_position(text_opinion, EntityEndType.Target)

        return pos1.get_index(TermPositionTypes.SentenceIndex) == \
               pos2.get_index(TermPositionTypes.SentenceIndex)

    # endregion

    # region public 'iter' methods

    @staticmethod
    def iter_frame_variants_with_indices_in_sentence(text_opinion):
        parsed_news, t_id = TextOpinionHelper.__get(text_opinion, EntityEndType.Target)
        s_index = TextOpinionHelper.extract_entity_sentence_index(text_opinion, EntityEndType.Source)
        return parsed_news.iter_sentence_terms(
            sentence_index=s_index,
            term_check=lambda term: isinstance(term, TextFrameVariant))

    # endregion

    # region private methods

    @staticmethod
    def __get_entity_position(text_opinion, end_type):
        parsed_news, _id = TextOpinionHelper.__get(text_opinion, end_type)
        return parsed_news.get_entity_position(_id)

    @staticmethod
    def __calc_distance_in_terms(pos1, pos2):
        assert(isinstance(pos1, TermPosition))
        assert(isinstance(pos2, TermPosition))

        return abs(pos1.get_index(TermPositionTypes.IndexInDocument) -
                   pos2.get_index(TermPositionTypes.IndexInDocument))

    @staticmethod
    def __get(text_opinion, end_type):
        owner = text_opinion.Owner
        assert(isinstance(owner, TextOpinionCollection))
        id = TextOpinionHelper.__get_end_id(text_opinion, end_type)
        pnc = owner.RelatedParsedNewsCollection
        assert(isinstance(pnc, ParsedNewsCollection))
        parsed_news = pnc.get_by_news_id(text_opinion.NewsID)
        return parsed_news, id

    @staticmethod
    def __get_end_id(text_opinion, end_type):
        assert(isinstance(text_opinion, TextOpinion))
        assert(end_type == EntityEndType.Source or end_type == EntityEndType.Target)
        return text_opinion.SourceId if end_type == EntityEndType.Source else text_opinion.TargetId

    # endregion
