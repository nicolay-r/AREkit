from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.linked_text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.contrib.bert.formatters.opinions.provider import OpinionProvider
from arekit.contrib.bert.formatters.row_ids.base import BaseIDFormatter


class BinaryIDFormatter(BaseIDFormatter):
    """
    Considered that label of opinion IS A PART OF id.
    """

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

        template = BaseIDFormatter.SEPARATOR.join([u"{multiple}", u"{label}"])

        return template.format(multiple=o_id,
                               label=label_scaler.label_to_uint(linked_opinions.get_linked_label()))

    @staticmethod
    def parse_label_in_sample_id(row_id):
        assert(isinstance(row_id, unicode))
        return int(row_id[row_id.index(u'l') + 1:len(row_id)])

    @staticmethod
    def parse_index_in_sample_id(row_id):
        assert(isinstance(row_id, unicode))
        return int(row_id[row_id.index(BaseIDFormatter.INDEX[0]) + 1:row_id.index(BaseIDFormatter.SEPARATOR)])

