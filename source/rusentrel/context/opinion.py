from arekit.common.labels.base import Label
from arekit.common.text_opinions.base import TextOpinion
from arekit.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection


class RuSentRelTextOpinion(TextOpinion):
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

        super(RuSentRelTextOpinion, self).__init__(
            news_id=rusentrel_news_id,
            source_id=e_source_doc_level_id,
            target_id=e_target_doc_level_id,
            label=sentiment,
            owner=None,
            text_opinion_id=None)
