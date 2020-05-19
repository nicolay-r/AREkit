from arekit.bert.providers.row_ids.base import BaseIDProvider


class MultipleIDProvider(BaseIDProvider):
    """
    Considered that label of opinion is not a part of id.
    """

    @staticmethod
    def create_sample_id(opinion_provider, linked_opinions, index_in_linked, label_scaler):
        return BaseIDProvider.create_opinion_id(opinion_provider=opinion_provider,
                                                linked_opinions=linked_opinions,
                                                index_in_linked=index_in_linked)
