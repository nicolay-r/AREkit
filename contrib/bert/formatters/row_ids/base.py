from arekit.common.linked_text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.contrib.bert.formatters.opinions.provider import OpinionProvider


class BaseIDFormatter(object):
    """
    Opinion in text is a sequence of opinions in context
    o1, o2, o3, ..., on

    o1 -- first_text_opinion
    i -- index in lined (for example: i=3 => 03)
    """

    @staticmethod
    def create_opinion_id(opinion_provider, linked_opinions, index_in_linked):
        assert(isinstance(opinion_provider, OpinionProvider))
        assert(isinstance(linked_opinions, LinkedTextOpinionsWrapper))
        assert(isinstance(index_in_linked, int))

        return u"n{}_o{}_i{}".format(linked_opinions.FirstOpinion.NewsID,
                                     linked_opinions.FirstOpinion.TextOpinionID,
                                     index_in_linked)

    @staticmethod
    def create_sample_id(opinion_provider, linked_opinions, index_in_linked):
        raise NotImplementedError()
