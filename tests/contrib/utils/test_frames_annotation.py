import unittest

from arekit.common.frames.text_variant import TextFrameVariant
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, \
    RuSentiFramesEffectLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from tests.contrib.source.labels import PositiveLabel
from tests.contrib.utils.labels import NegativeLabel


class TestFramesAnnotation(unittest.TestCase):

    @staticmethod
    def __create_frames_variants_collection():
        frames = RuSentiFramesCollection.read_collection(
            version=RuSentiFramesVersions.V20,
            labels_fmt=RuSentiFramesLabelsFormatter(
                neg_label_type=NegativeLabel, pos_label_type=PositiveLabel),
            effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(
                neg_label_type=NegativeLabel, pos_label_type=PositiveLabel))

        frame_variant_collection = FrameVariantsCollection()
        frame_variant_collection.fill_from_iterable(
            variants_with_id=frames.iter_frame_id_and_variants(),
            overwrite_existed_variant=True,
            raise_error_on_existed_variant=False)

        return frame_variant_collection

    def test(self):
        frame_variants_collection = self.__create_frames_variants_collection()
        stemmer = MystemWrapper()
        parser = LemmasBasedFrameVariantsParser(save_lemmas=False,
                                                stemmer=stemmer,
                                                frame_variants=frame_variants_collection)

        input_terms = "мы пытались его осудить но не получилось".split()

        for term in parser.apply(input_terms):
            str_term = "[{}]".format(term.Variant.get_value()) \
                if isinstance(term, TextFrameVariant) else term
            print(str_term)


if __name__ == '__main__':
    unittest.main()
