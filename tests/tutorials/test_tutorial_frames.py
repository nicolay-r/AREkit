import unittest

from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.labels.base import Label
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, \
    RuSentiFramesEffectLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.connotations.rusentiframes_sentiment import RuSentiFramesConnotationProvider


class Positive(Label):
    pass


class Negative(Label):
    pass


class TestFrames(unittest.TestCase):

    def test(self):
        frames_collection = RuSentiFramesCollection.read(
            version=RuSentiFramesVersions.V20,
            labels_fmt=RuSentiFramesLabelsFormatter(
                pos_label_type=Positive, neg_label_type=Negative),
            effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(
                pos_label_type=Positive, neg_label_type=Negative))

        frame_variant_collection = FrameVariantsCollection()
        frame_variant_collection.fill_from_iterable(
            variants_with_id=frames_collection.iter_frame_id_and_variants(),
            overwrite_existed_variant=True,
            raise_error_on_existed_variant=False)

        frames_connotation_provider = RuSentiFramesConnotationProvider(frames_collection)


if __name__ == '__main__':
    unittest.main()
