from arekit.common.frames.collection import FramesCollection
from arekit.common.frames.polarity import FramePolarity
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler import BaseLabelScaler
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.contrib.networks.features.utils import create_filled_array


class FrameRoleFeatures(object):

    @staticmethod
    def from_tsv(frame_variants, frames_collection, three_label_scaler):
        assert(isinstance(three_label_scaler, BaseLabelScaler))

        result = []

        for variant in frame_variants:

            value = FrameRoleFeatures.__extract_uint_frame_variant_sentiment_role(
                text_frame_variant=variant,
                frames_collection=frames_collection,
                three_label_scaler=three_label_scaler)

            result.append(value)

        return result

    @staticmethod
    def to_input(frame_inds, frame_sent_roles, size, filler):
        assert(isinstance(frame_inds, list))
        assert(isinstance(frame_sent_roles, list))
        assert(len(frame_inds) == len(frame_sent_roles))

        vector = create_filled_array(size=size, value=filler)

        for frame_ind, frame_ind_in_sample in enumerate(frame_inds):
            if frame_ind_in_sample >= len(vector):
                continue
            vector[frame_ind_in_sample] = frame_sent_roles[frame_ind]

        return vector

    # region private methods

    @staticmethod
    def __extract_uint_frame_variant_sentiment_role(text_frame_variant, frames_collection, three_label_scaler):
        assert(isinstance(text_frame_variant, TextFrameVariant))
        assert(isinstance(frames_collection, FramesCollection))
        assert(isinstance(three_label_scaler, BaseLabelScaler))

        frame_id = text_frame_variant.Variant.FrameID
        polarity = frames_collection.try_get_frame_sentiment_polarity(frame_id)

        if polarity is None:
            return three_label_scaler.label_to_uint(label=NoLabel())

        assert(isinstance(polarity, FramePolarity))

        if text_frame_variant.IsInverted:
            inv_label = three_label_scaler.invert_label(polarity.Label)
            return three_label_scaler.label_to_uint(label=inv_label)

        return three_label_scaler.label_to_uint(polarity.Label)

    # endregion
