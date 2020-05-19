from arekit.bert.providers.row_ids.base import BaseIDProvider
from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.bert.providers.opinions import OpinionProvider


class BinaryIDProvider(BaseIDProvider):
    """
    Considered that label of opinion IS A PART OF id.
    """

    LABEL = u'l{}' + BaseIDProvider.SEPARATOR

    @staticmethod
    def create_sample_id(opinion_provider, linked_opinions, index_in_linked, label_scaler):
        assert(isinstance(opinion_provider, OpinionProvider))
        assert(isinstance(linked_opinions, LinkedTextOpinionsWrapper))
        assert(isinstance(index_in_linked, int))
        assert(isinstance(label_scaler, BaseLabelScaler))

        o_id = BaseIDProvider.create_opinion_id(
            opinion_provider=opinion_provider,
            linked_opinions=linked_opinions,
            index_in_linked=index_in_linked)

        template = u''.join([u"{}", BinaryIDProvider.LABEL])

        return template.format(o_id,
                               label_scaler.label_to_uint(linked_opinions.get_linked_label()))

    @staticmethod
    def parse_label_in_sample_id(sample_id):
        assert(isinstance(sample_id, unicode))
        return BinaryIDProvider._parse(row_id=sample_id, pattern=BinaryIDProvider.LABEL)

    @staticmethod
    def parse_index_in_sample_id(sample_id):
        assert(isinstance(sample_id, unicode))
        return BinaryIDProvider._parse(row_id=sample_id, pattern=BinaryIDProvider.INDEX)

