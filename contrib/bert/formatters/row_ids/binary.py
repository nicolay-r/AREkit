from arekit.common.linked_text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.contrib.bert.formatters.opinions.provider import OpinionProvider
from arekit.contrib.bert.formatters.row_ids.base import BaseIDFormatter


class BinaryIDFormatter(BaseIDFormatter):
    """
    Considered that label of opinion IS A PART OF id.
    """

    @staticmethod
    def create_sample_id(opinion_provider, linked_opinions, index_in_linked):
        assert(isinstance(opinion_provider, OpinionProvider))
        assert(isinstance(linked_opinions, LinkedTextOpinionsWrapper))
        assert(isinstance(index_in_linked, int))

        o_id = BaseIDFormatter.create_opinion_id(
            opinion_provider=opinion_provider,
            linked_opinions=linked_opinions,
            index_in_linked=index_in_linked)

        return u"{multiple}_l{label}".format(
            multiple=o_id,
            label=linked_opinions.get_linked_sentiment())

    @staticmethod
    def parse_label_in_sample_id(row_id):
        assert(isinstance(row_id, unicode))
        return int(row_id[row_id.index(u'l') + 1:len(row_id)])

    @staticmethod
    def parse_index_in_sample_id(row_id):
        assert(isinstance(row_id, unicode))
        return int(row_id[row_id.index(BaseIDFormatter.INDEX[0]) + 1:row_id.index(u'_')])

