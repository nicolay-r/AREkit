from arekit.common.frames.connotations.descriptor import FrameConnotationDescriptor
from arekit.common.frames.connotations.provider import FrameConnotationProvider
from arekit.common.frames.text_variant import TextFrameVariant
from arekit.common.labels.scaler.sentiment import SentimentLabelScaler


def extract_uint_frame_variant_connotation(text_frame_variant, frames_connotation_provider, three_label_scaler):
    assert (isinstance(text_frame_variant, TextFrameVariant))
    assert (isinstance(frames_connotation_provider, FrameConnotationProvider))
    assert (isinstance(three_label_scaler, SentimentLabelScaler))

    frame_id = text_frame_variant.Variant.FrameID
    connot_descriptor = frames_connotation_provider.try_provide(frame_id)

    if connot_descriptor is None:
        return three_label_scaler.label_to_uint(label=three_label_scaler.get_no_label_instance())

    assert (isinstance(connot_descriptor, FrameConnotationDescriptor))

    target_label = three_label_scaler.invert_label(connot_descriptor.Label) \
        if text_frame_variant.IsNegated else connot_descriptor.Label

    return three_label_scaler.label_to_uint(target_label)
