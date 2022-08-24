from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.opinions.base import Opinion
from arekit.contrib.source.brat.relation import BratRelation
from arekit.contrib.source.ruattitudes.opinions.base import SentenceOpinion


class RuAttitudesSentenceOpinionConverter:

    @staticmethod
    def to_brat_relation(sentence_opinion, end_to_doc_id_func):
        """ Converts opinion into brat-related relation.
            NOTE: for rel_type we just call str() over int-based value.
        """
        assert(isinstance(sentence_opinion, SentenceOpinion))
        return BratRelation(id_in_doc="0",
                            source_id=end_to_doc_id_func(sentence_opinion.SourceID),
                            target_id=end_to_doc_id_func(sentence_opinion.TargetID),
                            rel_type=str(sentence_opinion.Label))

    @staticmethod
    def to_opinion(sentence_opinion, source_value, target_value, label_scaler):
        """
        Converts onto document, non referenced opinion
        (non bounded to the text).
        """
        assert(isinstance(sentence_opinion, SentenceOpinion))
        assert(isinstance(label_scaler, BaseLabelScaler))

        opinion = Opinion(source_value=source_value,
                          target_value=target_value,
                          sentiment=label_scaler.int_to_label(sentence_opinion.Label))

        # Using this tag allows to perform a revert operation,
        # i.e. to find opinion_ref by opinion.
        opinion.set_tag(sentence_opinion.Tag)

        return opinion
