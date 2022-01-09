from arekit.common.frames.connotations.descriptor import FrameConnotationDescriptor
from arekit.common.frames.connotations.provider import FrameConnotationProvider
from arekit.common.frames.text_variant import TextFrameVariant
from arekit.common.labels.scaler.sentiment import SentimentLabelScaler
from arekit.contrib.networks.features.utils import create_filled_array


class FrameConnotationFeatures(object):

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
    def extract_uint_frame_variant_connotation(text_frame_variant, frames_connotation_provider, three_label_scaler):
        assert(isinstance(text_frame_variant, TextFrameVariant))
        assert(isinstance(frames_connotation_provider, FrameConnotationProvider))
        assert(isinstance(three_label_scaler, SentimentLabelScaler))

        frame_id = text_frame_variant.Variant.FrameID
        connot_descriptor = frames_connotation_provider.try_provide(frame_id)

        if connot_descriptor is None:
            return three_label_scaler.label_to_uint(label=three_label_scaler.get_no_label_instance())

        assert(isinstance(connot_descriptor, FrameConnotationDescriptor))

        # TODO #217 -- remove IsInverted. (we perfrom labels inversion during text processing, via extra
        # pipeline element)
        target_label = three_label_scaler.invert_label(connot_descriptor.Label) \
            if text_frame_variant.IsNegated else connot_descriptor.Label

        return three_label_scaler.label_to_uint(target_label)
