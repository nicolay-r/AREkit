from arekit.common.linked_text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.contrib.bert.formatters.opinions.provider import OpinionProvider


class BaseIDFormatter(object):
    """
    Opinion in text is a sequence of opinions in context
    o1, o2, o3, ..., on

    o1 -- first_text_opinion
    i -- index in lined (for example: i=3 => 03)
    """

    # region 'create' methods

    @staticmethod
    def create_opinion_id(opinion_provider, linked_opinions, index_in_linked):
        assert(isinstance(opinion_provider, OpinionProvider))
        assert(isinstance(linked_opinions, LinkedTextOpinionsWrapper))
        assert(isinstance(index_in_linked, int))

        return u"n{news}_o{opinion}_i{index}".format(news=linked_opinions.FirstOpinion.NewsID,
                                                     opinion=linked_opinions.FirstOpinion.TextOpinionID,
                                                     index=index_in_linked)

    @staticmethod
    def create_sample_id(opinion_provider, linked_opinions, index_in_linked):
        raise NotImplementedError()

    @staticmethod
    def create_news_id_pattern(news_id):
        assert(isinstance(news_id, int))
        return u"n{news}_".format(news=news_id)

    @staticmethod
    def create_opinion_id_pattern(opinion_id):
        assert(isinstance(opinion_id, int))
        return u"o{opinion}_".format(opinion=opinion_id)

    # endregion

    # region 'parse' methods

    @staticmethod
    def parse_opinion_in_opinion_id(row_id):
        assert(isinstance(row_id, unicode))
        return int(row_id[row_id.index(u'o') + 1:row_id.index(u'_')])

    @staticmethod
    def parse_news_in_sample_id(row_id):
        assert(isinstance(row_id, unicode))
        return int(row_id[row_id.index(u'n') + 1:row_id.index(u'_')])

    # endregion
