from arekit.common.labels.base import Label
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.contrib.bert.formatters.row_ids.multiple import MultipleIDFormatter


class BinaryIDFormatter(MultipleIDFormatter):
    """
    Considered that label of opinion IS A PART OF id.
    """

    @staticmethod
    def create_opinion_id(first_text_opinion, index_in_linked):
        assert(isinstance(first_text_opinion, TextOpinion))

        o_id = MultipleIDFormatter.create_opinion_id(
            first_text_opinion=first_text_opinion,
            index_in_linked=index_in_linked)

        label = first_text_opinion.Sentiment

        assert(isinstance(label, Label))

        return u"{multiple}_l{label}".format(multiple=o_id,
                                             label=label.to_uint())

