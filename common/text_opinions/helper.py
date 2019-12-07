from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text_opinions.collection import TextOpinionCollection
from arekit.common.text_opinions.base import TextOpinion
from arekit.common.text_opinions.end_type import EntityEndType


class TextOpinionHelper(object):
    """
    This class provides a helper functions for TextOpinions, which become a part of TextOpinionCollection.
    The latter is important because of the dependency from Owner.
    We utilize 'extract' prefix in methods to emphasize that these are methods of helper.
    """

    # region public 'extract' methods

    @staticmethod
    def extract_entity_value(text_opinion, end_type):
        parsed_news, id = TextOpinionHelper.__get(text_opinion, end_type)
        return parsed_news.get_entity_value(id)

    @staticmethod
    def extract_entity_sentence_index(text_opinion, end_type):
        parsed_news, id = TextOpinionHelper.__get(text_opinion, end_type)
        return parsed_news.get_entity_sentence_index(id)

    @staticmethod
    def extract_entity_doc_level_term_index(text_opinion, end_type):
        return TextOpinionHelper.__entity_document_level_term_index(text_opinion, end_type)

    @staticmethod
    def extract_entity_sentence_level_term_index(text_opinion, end_type):
        parsed_news, id = TextOpinionHelper.__get(text_opinion, end_type)
        return parsed_news.get_entity_sentence_level_term_index(id)

    @staticmethod
    def extract_entity_sentence_level_synonym_indices(text_opinion, end_type, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))
        parsed_news, id = TextOpinionHelper.__get(text_opinion, end_type)
        s_index = parsed_news.get_entity_sentence_index(id)
        e_value = parsed_news.get_entity_value(id)
        e_group = synonyms.get_synonym_group_index(e_value)

        inds = []

        for e_index, e in parsed_news.iter_sentence_entities_with_indices(s_index):

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
    def calculate_distance_between_entities_in_terms(text_opinion):
        return abs(TextOpinionHelper.__entity_document_level_term_index(text_opinion, EntityEndType.Source) -
                   TextOpinionHelper.__entity_document_level_term_index(text_opinion, EntityEndType.Target))

    # endregion

    # region public 'check' methods

    @staticmethod
    def check_ends_has_same_sentence_index(text_opinion):
        _, s_id = TextOpinionHelper.__get(text_opinion, EntityEndType.Source)
        parsed_news, t_id = TextOpinionHelper.__get(text_opinion, EntityEndType.Target)
        return parsed_news.get_entity_sentence_index(s_id) == \
               parsed_news.get_entity_sentence_index(t_id)

    # endregion

    # region public 'iter' methods

    @staticmethod
    def iter_frame_variants_with_indices_in_sentence(text_opinion):
        parsed_news, t_id = TextOpinionHelper.__get(text_opinion, EntityEndType.Target)
        s_index = TextOpinionHelper.extract_entity_sentence_index(text_opinion, EntityEndType.Source)
        return parsed_news.iter_sentence_frame_variants_with_indices(sentence_index=s_index)

    # endregion

    # region private methods

    @staticmethod
    def __entity_document_level_term_index(text_opinion, end_type):
        parsed_news, id = TextOpinionHelper.__get(text_opinion, end_type)
        return parsed_news.get_entity_document_level_term_index(id)

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
