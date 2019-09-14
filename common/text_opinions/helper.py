from core.common.parsed_news.collection import ParsedNewsCollection
from core.common.text_opinions.text_opinion import TextOpinion
from core.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from core.common.text_opinions.end_type import EntityEndType


class TextOpinionHelper(object):

    @staticmethod
    def EntityValue(text_opinion, end_type):
        parsed_news, id = TextOpinionHelper.__get(text_opinion, end_type)
        return parsed_news.get_entity_value(id)

    @staticmethod
    def EntitySentenceIndex(text_opinion, end_type):
        parsed_news, id = TextOpinionHelper.__get(text_opinion, end_type)
        return parsed_news.get_entity_sentence_index(id)

    @staticmethod
    def CheckEndsHasSameSentenceIndex(text_opinion):
        _, s_id = TextOpinionHelper.__get(text_opinion, EntityEndType.Source)
        parsed_news, t_id = TextOpinionHelper.__get(text_opinion, EntityEndType.Target)
        return parsed_news.get_entity_sentence_index(s_id) ==\
               parsed_news.get_entity_sentence_index(t_id)

    @staticmethod
    def EntityDocumentLevelTermIndex(text_opinion, end_type):
        return TextOpinionHelper.__entity_document_level_term_index(text_opinion, end_type)

    @staticmethod
    def EntitySentenceLevelTermIndex(text_opinion, end_type):
        parsed_news, id = TextOpinionHelper.__get(text_opinion, end_type)
        return parsed_news.get_entity_sentence_level_term_index(id)

    @staticmethod
    def DistanceBetweenEntitiesInTerms(text_opinion):
        return abs(TextOpinionHelper.__entity_document_level_term_index(text_opinion, EntityEndType.Source) -
                   TextOpinionHelper.__entity_document_level_term_index(text_opinion, EntityEndType.Target))

    @staticmethod
    def __entity_document_level_term_index(text_opinion, end_type):
        parsed_news, id = TextOpinionHelper.__get(text_opinion, end_type)
        return parsed_news.get_entity_document_level_term_index(id)

    @staticmethod
    def __get(text_opinion, end_type):
        owner = text_opinion.Owner
        # TODO. Fix it: TextOpinionCollection instead of LabeledLinked...
        assert(isinstance(owner, LabeledLinkedTextOpinionCollection))
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
