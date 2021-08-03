from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper


class BaseIDProvider(object):
    """
    Opinion in text is a sequence of opinions in context
    o1, o2, o3, ..., on

    o1 -- first_text_opinion
    i -- index in lined (for example: i=3 => 03)
    """

    SEPARATOR = u'_'
    OPINION = u"o{}" + SEPARATOR
    INDEX = u"i{}" + SEPARATOR

    # region 'create' methods

    @staticmethod
    def create_opinion_id(linked_opinions, index_in_linked):
        assert(isinstance(linked_opinions, LinkedTextOpinionsWrapper))
        assert(isinstance(index_in_linked, int))

        template = u''.join([BaseIDProvider.OPINION,
                             BaseIDProvider.INDEX])

        text_opinion_id = linked_opinions.First.TextOpinionID
        assert(isinstance(text_opinion_id, int))

        return template.format(text_opinion_id,
                               index_in_linked)

    @staticmethod
    def create_sample_id(linked_opinions, index_in_linked, label_scaler):
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
        return sample_id[:sample_id.index(BaseIDProvider.INDEX[0])] + BaseIDProvider.INDEX.format(0)

    # region 'parse' methods

    @staticmethod
    def _parse(row_id, pattern):
        assert(isinstance(pattern, unicode))

        _from = row_id.index(pattern[0]) + 1
        _to = row_id.index(BaseIDProvider.SEPARATOR, _from, len(row_id))

        return int(row_id[_from:_to])

    @staticmethod
    def parse_opinion_in_opinion_id(opinion_id):
        assert(isinstance(opinion_id, unicode))
        return BaseIDProvider._parse(opinion_id, BaseIDProvider.OPINION)

    @staticmethod
    def parse_opinion_in_sample_id(sample_id):
        assert(isinstance(sample_id, unicode))
        return BaseIDProvider._parse(sample_id, BaseIDProvider.OPINION)

    # endregion
