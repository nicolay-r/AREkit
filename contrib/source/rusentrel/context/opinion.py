from arekit.common.labels.base import Label
from arekit.common.text_opinions.base import TextOpinion


class RuSentRelTextOpinion(TextOpinion):
    """
    Strict Relation between two Entities
    """

    def __init__(self,
                 rusentrel_news_id,
                 e_source_doc_level_id,
                 e_target_doc_level_id,
                 sentiment):
        assert(isinstance(rusentrel_news_id, int))
        assert(isinstance(e_source_doc_level_id, int))
        assert(isinstance(e_target_doc_level_id, int))
        assert(isinstance(sentiment, Label))

        super(RuSentRelTextOpinion, self).__init__(
            news_id=rusentrel_news_id,
            source_id=e_source_doc_level_id,
            target_id=e_target_doc_level_id,
            label=sentiment,
            owner=None,
            text_opinion_id=None)
