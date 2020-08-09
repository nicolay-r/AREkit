import unittest
from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.io_utils import RuSentiFramesVersions
from arekit.processing.lemmatization.mystem import MystemWrapper


class TestRuSentiFrameVariants(unittest.TestCase):

    def __iter_frame_variants(self):
        stemmer = MystemWrapper()
        frames_collection = RuSentiFramesCollection.read_collection(RuSentiFramesVersions.V20)
        frame_variants = FrameVariantsCollection.create_unique_variants_from_iterable(
            variants_with_id=frames_collection.iter_frame_id_and_variants(),
            stemmer=stemmer)

        for v, _ in frame_variants.iter_variants():
            yield v

    def test_iter_frame_variants(self):
        frame_values_list = list(self.__iter_frame_variants())
        for frame_variant in frame_values_list:
            print u'"{}"'.format(frame_variant)


if __name__ == '__main__':
    unittest.main()
