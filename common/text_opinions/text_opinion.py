from core.common.ref_opinon import RefOpinion
from core.evaluation.labels import Label


class TextOpinion(RefOpinion):
    """
    Represents a relation which were found in news article
    and composed between two named entities
        (it was found especially by Opinion with predefined label)
        allows to modify label using set_label
    """

    # region constructors

    def __init__(self, news_id, text_opinion_id, source_id, target_id, owner, label):
        assert(isinstance(news_id, int))
        assert(isinstance(text_opinion_id, int) or text_opinion_id is None)

        super(TextOpinion, self).__init__(source_id=source_id,
                                          target_id=target_id,
                                          sentiment=label,
                                          owner=owner)
        self.__news_id = news_id
        self.__text_opinion_id = text_opinion_id
        self.__label = label

    @classmethod
    def create_copy(cls, other):
        assert(isinstance(other, TextOpinion))
        return TextOpinion(news_id=other.__news_id,
                           text_opinion_id=other.__text_opinion_id,
                           source_id=other.SourceId,
                           target_id=other.TargetId,
                           owner=other.Owner,
                           label=other.Sentiment)

    @classmethod
    def create_from_ref_opinion(cls, news_id, text_opinion_id, ref_opinion):
        assert(isinstance(ref_opinion, RefOpinion))
        return cls(news_id=news_id,
                   text_opinion_id=text_opinion_id,
                   source_id=ref_opinion.SourceId,
                   target_id=ref_opinion.TargetId,
                   owner=ref_opinion.Owner,
                   label=ref_opinion.Sentiment)

    # endregion

    # region properties

    @property
    def Sentiment(self):
        return self.__label

    @property
    def NewsID(self):
        return self.__news_id

    @property
    def TextOpinionID(self):
        return self.__text_opinion_id

    # endregion

    # region public methods

    def set_text_opinion_id(self, relation_id):
        assert(isinstance(relation_id, int))
        self.__text_opinion_id = relation_id

    def set_label(self, label):
        assert(isinstance(label, Label))
        self.__label = label

    # endregion
