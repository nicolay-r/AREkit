from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.frames.collection import FramesCollection
from arekit.common.frames.polarity import FramePolarity
from arekit.common.labels.base import NeutralLabel
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.common.text_opinions.helper import TextOpinionHelper


class FrameFeatures(object):

    @staticmethod
    def compose_frames(text_opinion):
        frames = []
        # TODO. Duplicates iteration (FIX)
        indices = list(TextOpinionHelper.iter_frame_variants_with_indices_in_sentence(text_opinion))
        # NOTE: We utilize in reverse mode to prevent a case of zero-based frame by the beginning.
        for index, _ in reversed(indices):
            frames.append(index)
        return frames


    @staticmethod
    def compose_frame_roles(text_opinion, size, frames_collection, filler, label_scaler):
        assert(isinstance(label_scaler, BaseLabelScaler))

        result = [filler] * size

        # TODO. Duplicates iteration (FIX)
        for index, variant in TextOpinionHelper.iter_frame_variants_with_indices_in_sentence(text_opinion):

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
