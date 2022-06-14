from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.opinions.base import Opinion
from arekit.common.text_opinions.base import TextOpinion
from arekit.contrib.source.ruattitudes.opinions.base import SentenceOpinion


class RuAttitudesSentenceOpinionConverter:

    @staticmethod
    def to_text_opinion(sentence_opinion, doc_id, end_to_doc_id_func, text_opinion_id, label_scaler):
        """
        Converts opinion into document-level referenced opinion
        """
        assert(isinstance(sentence_opinion, SentenceOpinion))
        assert (isinstance(label_scaler, BaseLabelScaler))

        return TextOpinion(doc_id=doc_id,
                           text_opinion_id=text_opinion_id,
                           source_id=end_to_doc_id_func(sentence_opinion.SourceID),
                           target_id=end_to_doc_id_func(sentence_opinion.TargetID),
                           owner=None,
                           label=label_scaler.int_to_label(sentence_opinion.Label))

    @staticmethod
    def to_opinion(sentence_opinion, source_value, target_value, label_scaler):
        """
        Converts onto document, non referenced opinion
        (non bounded to the text).
        """
        assert(isinstance(sentence_opinion, SentenceOpinion))
        assert (isinstance(label_scaler, BaseLabelScaler))

        opinion = Opinion(source_value=source_value,
                          target_value=target_value,
                          sentiment=label_scaler.int_to_label(sentence_opinion.Label))

        # Using this tag allows to perform a revert operation,
        # i.e. to find opinion_ref by opinion.
        opinion.set_tag(sentence_opinion.Tag)

        return opinion
