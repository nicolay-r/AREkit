from arekit.common.linked_text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.contrib.bert.formatters.opinions.provider import OpinionProvider


class BaseIDFormatter(object):
    """
    Opinion in text is a sequence of opinions in context
    o1, o2, o3, ..., on

    o1 -- first_text_opinion
    i -- index in lined (for example: i=3 => 03)
    """

    SEPARATOR = u'_'
    NEWS = u"n{}" + SEPARATOR
    OPINION = u"o{}" + SEPARATOR
    INDEX = u"i{}" + SEPARATOR

    # region 'create' methods

    @staticmethod
    def create_opinion_id(opinion_provider, linked_opinions, index_in_linked):
        assert(isinstance(opinion_provider, OpinionProvider))
        assert(isinstance(linked_opinions, LinkedTextOpinionsWrapper))
        assert(isinstance(index_in_linked, int))

        template = u''.join([BaseIDFormatter.NEWS,
                             BaseIDFormatter.OPINION,
                             BaseIDFormatter.INDEX])

        return template.format(linked_opinions.FirstOpinion.NewsID,
                               linked_opinions.FirstOpinion.TextOpinionID,
                               index_in_linked)

    @staticmethod
    def create_sample_id(opinion_provider, linked_opinions, index_in_linked, label_scaler):
        raise NotImplementedError()

    @staticmethod
    def create_pattern(id_value, p_type):
        assert(isinstance(id_value, int))
        assert(isinstance(p_type, unicode))
        return p_type.format(id_value)

    # endregion

    @staticmethod
    def convert_sample_id_to_opinion_id(sample_id):
        assert(isinstance(sample_id, unicode))
        return sample_id[:sample_id.index(BaseIDFormatter.INDEX[0])] + BaseIDFormatter.INDEX.format(0)

    # region 'parse' methods

    @staticmethod
    def _parse(row_id, pattern):
        assert(isinstance(pattern, unicode))

        _from = row_id.index(pattern[0]) + 1
        _to = row_id.index(BaseIDFormatter.SEPARATOR, _from, len(row_id))

        return int(row_id[_from:_to])

    @staticmethod
    def parse_opinion_in_opinion_id(opinion_id):
        assert(isinstance(opinion_id, unicode))
        return BaseIDFormatter._parse(opinion_id, BaseIDFormatter.OPINION)

    @staticmethod
    def parse_news_in_sample_id(opinion_id):
        assert(isinstance(opinion_id, unicode))
        return BaseIDFormatter._parse(opinion_id, BaseIDFormatter.NEWS)

    # endregion
