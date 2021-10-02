from arekit.common.data.row_ids.base import BaseIDProvider
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.labels.scaler import BaseLabelScaler


class BinaryIDProvider(BaseIDProvider):
    """
    Considered that label of opinion IS A PART OF id.
    """

    LABEL = 'l{}' + BaseIDProvider.SEPARATOR

    @staticmethod
    def create_sample_id(linked_opinions, index_in_linked, label_scaler):
        assert(isinstance(linked_opinions, LinkedTextOpinionsWrapper))
        assert(isinstance(index_in_linked, int))
        assert(isinstance(label_scaler, BaseLabelScaler))

        o_id = BaseIDProvider.create_opinion_id(linked_opinions=linked_opinions,
                                                index_in_linked=index_in_linked)

        template = ''.join(["{}", BinaryIDProvider.LABEL])

        return template.format(o_id,
                               label_scaler.label_to_uint(linked_opinions.get_linked_label()))

    @staticmethod
    def parse_label_in_sample_id(sample_id):
        assert(isinstance(sample_id, str))
        return BinaryIDProvider._parse(row_id=sample_id, pattern=BinaryIDProvider.LABEL)

    @staticmethod
    def parse_index_in_sample_id(sample_id):
        assert(isinstance(sample_id, str))
        return BinaryIDProvider._parse(row_id=sample_id, pattern=BinaryIDProvider.INDEX)

