from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.frames.collection import FramesCollection
from arekit.common.frames.polarity import FramePolarity
from arekit.common.labels.base import NeutralLabel
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.common.text_opinions.helper import TextOpinionHelper


class FrameFeatures(object):

    def __init__(self, text_opinion_helper):
        assert(isinstance(text_opinion_helper, TextOpinionHelper))
        self.__text_opinion_helper = text_opinion_helper

    def __iter_frame_variants(self, text_opinion):
        it = self.__text_opinion_helper.iter_terms_in_related_sentence(
            text_opinion=text_opinion,
            return_ind_in_sent=True,
            term_check=lambda term: isinstance(term, TextFrameVariant))

        for s_ind, frame_variant in it:
            yield s_ind, frame_variant

    def compose_frames(self, text_opinion):
        """
        NOTE: We utilize in reverse mode to prevent a case of zero-based frame by the beginning.
        """
        frames = []
        indices = list(self.__iter_frame_variants(text_opinion))

        for index, _ in reversed(indices):
            frames.append(index)

        return frames

    def compose_frame_roles(self, text_opinion, size, frames_collection, filler, label_scaler):
        assert(isinstance(label_scaler, BaseLabelScaler))

        result = [filler] * size

        for index, variant in self.__iter_frame_variants(text_opinion):

            if index >= len(result):
                continue

            value = FrameFeatures.__extract_uint_frame_variant_sentiment_role(
                text_frame_variant=variant,
                frames_collection=frames_collection,
                label_scaler=label_scaler)

            result[index] = value

        return result

    # region private methods

    @staticmethod
    def __extract_uint_frame_variant_sentiment_role(text_frame_variant, frames_collection, label_scaler):
        assert(isinstance(text_frame_variant, TextFrameVariant))
        assert(isinstance(frames_collection, FramesCollection))
        assert(isinstance(label_scaler, BaseLabelScaler))

        frame_id = text_frame_variant.Variant.FrameID
        polarity = frames_collection.try_get_frame_sentiment_polarity(frame_id)

        if polarity is None:
            return label_scaler.label_to_uint(label=NeutralLabel())

        assert(isinstance(polarity, FramePolarity))

        if text_frame_variant.IsInverted:
            inv_label = label_scaler.invert_label(polarity.Label)
            return label_scaler.label_to_uint(label=inv_label)

        return label_scaler.label_to_uint(polarity.Label)

    # endregion
