from arekit.contrib.bert.formatters.row_ids.base import BaseIDFormatter


class MultipleIDFormatter(BaseIDFormatter):
    """
    Considered that label of opinion is not a part of id.
    """

    @staticmethod
    def create_sample_id(opinion_provider, linked_opinions, index_in_linked, label_scaler):
        return BaseIDFormatter.create_opinion_id(opinion_provider=opinion_provider,
                                                 linked_opinions=linked_opinions,
                                                 index_in_linked=index_in_linked)
