from arekit.common.data.row_ids.base import BaseIDProvider


class MultipleIDProvider(BaseIDProvider):
    """
    Considered that label of opinion is not a part of id.
    # TODO. #376 related. This should be removed after refactoring, because
    # TODO. we consider an ordinary IDs, that not based on the other data.
    """

    @staticmethod
    def create_sample_id(linked_opinions, index_in_linked, label_scaler):
        return BaseIDProvider.create_opinion_id(text_opinions_linkage=linked_opinions,
                                                index_in_linked=index_in_linked)
