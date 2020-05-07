from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.linked_text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.contrib.bert.formatters.opinions.provider import OpinionProvider
from arekit.contrib.bert.formatters.row_ids.base import BaseIDFormatter


class BinaryIDFormatter(BaseIDFormatter):
    """
    Considered that label of opinion IS A PART OF id.
    """

    Label = u'l'

    @staticmethod
    def create_sample_id(opinion_provider, linked_opinions, index_in_linked, label_scaler):
        assert(isinstance(opinion_provider, OpinionProvider))
        assert(isinstance(linked_opinions, LinkedTextOpinionsWrapper))
        assert(isinstance(index_in_linked, int))
        assert(isinstance(label_scaler, BaseLabelScaler))

        o_id = BaseIDFormatter.create_opinion_id(
            opinion_provider=opinion_provider,
            linked_opinions=linked_opinions,
            index_in_linked=index_in_linked)

        template = BaseIDFormatter.SEPARATOR.join([u"{multiple}",
                                                   BinaryIDFormatter.Label + u"{label}"])

        return template.format(multiple=o_id,
                               label=label_scaler.label_to_uint(linked_opinions.get_linked_label()))

    @staticmethod
    def create_index_id_pattern(index_id):
        assert(isinstance(index_id, int))
        return BaseIDFormatter.INDEX.format(index=index_id) + BaseIDFormatter.SEPARATOR

    @staticmethod
    def convert_sample_id_to_opinion_id(sample_id):
        return sample_id[:sample_id.index(BinaryIDFormatter.Label)]

    @staticmethod
    def parse_label_in_sample_id(sample_id):
        assert(isinstance(sample_id, unicode))
        return int(sample_id[sample_id.index(BinaryIDFormatter.Label) + 1:len(sample_id)])

    @staticmethod
    def parse_index_in_sample_id(sample_id):
        assert(isinstance(sample_id, unicode))
        return int(sample_id[sample_id.index(BaseIDFormatter.INDEX[0]) + 1:sample_id.index(BaseIDFormatter.SEPARATOR)])

