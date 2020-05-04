from arekit.common.linked_text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.contrib.bert.formatters.opinions.provider import OpinionProvider


class BaseIDFormatter(object):
    """
    Opinion in text is a sequence of opinions in context
    o1, o2, o3, ..., on

    o1 -- first_text_opinion
    i -- index in lined (for example: i=3 => 03)
    """

    NEWS = u"n{news}"
    OPINION = u"o{opinion}"
    INDEX = u"i{index}"
    SEPARATOR = u'_'

    # region 'create' methods

    @staticmethod
    def create_opinion_id(opinion_provider, linked_opinions, index_in_linked):
        assert(isinstance(opinion_provider, OpinionProvider))
        assert(isinstance(linked_opinions, LinkedTextOpinionsWrapper))
        assert(isinstance(index_in_linked, int))

        template = BaseIDFormatter.SEPARATOR.join([BaseIDFormatter.NEWS,
                                                   BaseIDFormatter.OPINION,
                                                   BaseIDFormatter.INDEX])

        return template.format(news=linked_opinions.FirstOpinion.NewsID,
                               opinion=linked_opinions.FirstOpinion.TextOpinionID,
                               index=index_in_linked)

    @staticmethod
    def create_sample_id(opinion_provider, linked_opinions, index_in_linked, label_scaler):
        raise NotImplementedError()

    @staticmethod
    def create_news_id_pattern(news_id):
        assert(isinstance(news_id, int))
        return BaseIDFormatter.NEWS.format(news=news_id)

    @staticmethod
    def create_opinion_id_pattern(opinion_id):
        assert(isinstance(opinion_id, int))
        return BaseIDFormatter.OPINION.format(opinion=opinion_id)

    @staticmethod
    def create_index_id_pattern(index_id):
        assert(isinstance(index_id, int))
        return BaseIDFormatter.INDEX.format(index=index_id)

    # endregion

    @staticmethod
    def convert_sample_id_to_opinion_id(sample_id):
        raise NotImplementedError()

    # region 'parse' methods

    @staticmethod
    def parse_opinion_in_opinion_id(row_id):
        assert(isinstance(row_id, unicode))
        return int(row_id[row_id.index(BaseIDFormatter.OPINION[0]) + 1:row_id.index(BaseIDFormatter.SEPARATOR)])

    @staticmethod
    def parse_news_in_sample_id(row_id):
        assert(isinstance(row_id, unicode))
        return int(row_id[row_id.index(BaseIDFormatter.NEWS[0]) + 1:row_id.index(BaseIDFormatter.SEPARATOR)])

    # endregion
