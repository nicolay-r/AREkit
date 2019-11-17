from core.common.frames.collection import FramesCollection
from core.common.frames.polarity import FramePolarity
from core.common.labels.base import NeutralLabel
from core.common.text_frame_variant import TextFrameVariant
from core.common.text_opinions.helper import TextOpinionHelper


def compose_frames(text_opinion):
    frames = []
    indices = list(TextOpinionHelper.iter_frame_variants_with_indices_in_sentence(text_opinion))
    # NOTE: We utilize in reverse mode to prevent a case of zero-based frame by the beginning.
    for index, frame in reversed(indices):
        frames.append(frame)
    return frames


def compose_frame_roles(text_opinion, size, frames_collection, filler):

    result = [filler] * size

    for index, variant in TextOpinionHelper.iter_frame_variants_with_indices_in_sentence(text_opinion):

        if index >= len(result):
            continue

        value = __extract_uint_frame_variant_sentiment_role(
            text_frame_variant=variant,
            frames_collection=frames_collection)

        result[index] = value

    return result


def __extract_uint_frame_variant_sentiment_role(text_frame_variant, frames_collection):
    assert(isinstance(text_frame_variant, TextFrameVariant))
    assert(isinstance(frames_collection, FramesCollection))
    frame_id = text_frame_variant.Variant.FrameID
    polarity = frames_collection.try_get_frame_sentiment_polarity(frame_id)
    if polarity is None:
        return NeutralLabel().to_uint()

    assert(isinstance(polarity, FramePolarity))

    return polarity.Label.to_uint()
