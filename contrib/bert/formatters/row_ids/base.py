from arekit.common.text_opinions.text_opinion import TextOpinion


class BaseIDFormatter(object):
    """
    Opinion in text is a sequence of opinions in context
    o1, o2, o3, ..., on

    o1 -- first_text_opinion
    i -- index in lined (for example: i=3 => 03)
    """

    @staticmethod
    def create_opinion_id(first_text_opinion, index_in_linked):
        assert(isinstance(first_text_opinion, TextOpinion))
        assert(isinstance(index_in_linked, int))

        return u"n{}_o{}_i{}".format(first_text_opinion.NewsID,
                                     first_text_opinion.TextOpinionID,
                                     index_in_linked)

    @staticmethod
    def create_sample_id(first_text_opinion, index_in_linked):
        raise NotImplementedError()
