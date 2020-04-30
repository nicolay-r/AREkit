from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.contrib.bert.formatters.row_ids.base import BaseIDFormatter


class MultipleIDFormatter(BaseIDFormatter):
    """
    Considered that label of opinion is not a part of id.
    """

    @staticmethod
    def create_opinion_id(first_text_opinion, index_in_linked):
        assert(isinstance(first_text_opinion, TextOpinion))
        assert(isinstance(index_in_linked, int))

        return u"n{}_o{}_i{}".format(first_text_opinion.NewsID,
                                     first_text_opinion.TextOpinionID,
                                     index_in_linked)

    @staticmethod
    def parse_news_id(row_id):
        assert(isinstance(row_id, unicode))
        return int(row_id[row_id.index(u'n') + 1:row_id.index(u'_')])

    @staticmethod
    def sample_row_id_to_opinion_id(row_id):
        """
        Id in sample rows has information of linked opinions.
        Here the latter ommited and id could be suffixed with 'i0' only.
        """
        assert(isinstance(row_id, unicode))
        return row_id[:row_id.find(u'i')] + u"i0"

