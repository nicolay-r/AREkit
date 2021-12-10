from arekit.common.frames.connotations.descriptor import FrameConnotationDescriptor
from arekit.common.frames.connotations.provider import FrameConnotationProvider
from arekit.common.frames.text_variant import TextFrameVariant
from arekit.common.labels.scaler import BaseLabelScaler
from arekit.contrib.networks.features.utils import create_filled_array


class FrameRoleFeatures(object):

    @ staticmethod
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

    @staticmethod
    def extract_uint_frame_variant_sentiment_role(text_frame_variant, frames_connotation_provider, three_label_scaler):
        assert(isinstance(text_frame_variant, TextFrameVariant))
        assert(isinstance(frames_connotation_provider, FrameConnotationProvider))
        assert(isinstance(three_label_scaler, BaseLabelScaler))

        frame_id = text_frame_variant.Variant.FrameID
        polarity = frames_connotation_provider.try_get_frame_sentiment_polarity(frame_id)

        if polarity is None:
            return three_label_scaler.label_to_uint(label=three_label_scaler.get_no_label_instance())

        assert(isinstance(polarity, FrameConnotationDescriptor))

        if text_frame_variant.IsInverted:
            inv_label = three_label_scaler.invert_label(polarity.Label)
            return three_label_scaler.label_to_uint(label=inv_label)

        return three_label_scaler.label_to_uint(polarity.Label)
