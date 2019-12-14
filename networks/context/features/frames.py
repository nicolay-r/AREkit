from arekit.common.frames.collection import FramesCollection
from arekit.common.frames.polarity import FramePolarity
from arekit.common.labels.base import NeutralLabel, PositiveLabel, NegativeLabel
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.common.text_opinions.helper import TextOpinionHelper


# TODO. Refactoring as static method.


def compose_frames(text_opinion):
    frames = []
    indices = list(TextOpinionHelper.iter_frame_variants_with_indices_in_sentence(text_opinion))
    # NOTE: We utilize in reverse mode to prevent a case of zero-based frame by the beginning.
    for index, _ in reversed(indices):
        frames.append(index)
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

    if text_frame_variant.IsInverted:
        return __create_inverted_label(polarity.Label).to_uint()

    return polarity.Label.to_uint()


def __create_inverted_label(label):
    if isinstance(label, NeutralLabel):
        return label
    if isinstance(label, NegativeLabel):
        return PositiveLabel()
    if isinstance(label, PositiveLabel):
        return NegativeLabel()

    return None
