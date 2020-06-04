from arekit.common.text_opinions.base import RefOpinion
from arekit.common.labels.base import Label
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection


class RuSentRelTextOpinion(RefOpinion):
    """
    Strict Relation between two Entities
    """

    def __init__(self,
                 rusentrel_news_id,
                 e_source_doc_level_id,
                 e_target_doc_level_id,
                 sentiment,
                 doc_entities):
        assert(isinstance(rusentrel_news_id, int))
        assert(isinstance(e_source_doc_level_id, int))
        assert(isinstance(e_target_doc_level_id, int))
        assert(isinstance(sentiment, Label))
        assert(isinstance(doc_entities, RuSentRelDocumentEntityCollection))
        super(RuSentRelTextOpinion, self).__init__(source_id=e_source_doc_level_id,
                                                   target_id=e_target_doc_level_id,
                                                   sentiment=sentiment,
                                                   owner=doc_entities)

        self.__news_id = rusentrel_news_id
        self.__entity_source_ID = e_source_doc_level_id
        self.__entity_target_ID = e_target_doc_level_id
        self.__entity_by_id_func = doc_entities.get_entity_by_id

    def to_text_opinion(self):
        return TextOpinion(
            news_id=self.RuSentRelNewsId,
            source_id=self.SourceId,
            target_id=self.TargetId,
            label=self.Sentiment,
            owner=None,
            text_opinion_id=None)

    # region properties

    @property
    def RuSentRelNewsId(self):
        return self.__news_id

    @property
    def SourceEntity(self):
        return self.__entity_by_id_func(self.__entity_source_ID)

    @property
    def TargetEntity(self):
        return self.__entity_by_id_func(self.__entity_target_ID)

    #endregion
