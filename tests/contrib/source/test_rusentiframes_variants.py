import sys
import unittest

sys.path.append('../../../../')

from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesEffectLabelsFormatter, \
    RuSentiFramesLabelsFormatter

from tests.contrib.source.labels import PositiveLabel, NegativeLabel


class TestRuSentiFrameVariants(unittest.TestCase):

    @staticmethod
    def __iter_frame_variants():
        frames_collection = RuSentiFramesCollection.read_collection(
            version=RuSentiFramesVersions.V20,
            labels_fmt=RuSentiFramesLabelsFormatter(
                neg_label_type=NegativeLabel, pos_label_type=PositiveLabel),
            effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(
                neg_label_type=NegativeLabel, pos_label_type=PositiveLabel))

        frame_variants = FrameVariantsCollection()
        frame_variants.fill_from_iterable(variants_with_id=frames_collection.iter_frame_id_and_variants(),
                                          overwrite_existed_variant=True,
                                          raise_error_on_existed_variant=False)

        for v, _ in frame_variants.iter_variants():
            yield v

    def test_iter_frame_variants(self):
        frame_values_list = list(self.__iter_frame_variants())
        for frame_variant in frame_values_list:
            print('"{}"'.format(frame_variant))


if __name__ == '__main__':
    unittest.main()
