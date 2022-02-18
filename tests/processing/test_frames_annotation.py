import unittest

from arekit.common.frames.text_variant import TextFrameVariant
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.text.pipeline_frames_lemmatized import LemmasBasedFrameVariantsParser


class TestFramesAnnotation(unittest.TestCase):

    @staticmethod
    def __create_frames_variants_collection():
        frames = RuSentiFramesCollection.read_collection(RuSentiFramesVersions.V20)
        frame_variant_collection = FrameVariantsCollection()
        frame_variant_collection.fill_from_iterable(
            variants_with_id=frames.iter_frame_id_and_variants(),
            overwrite_existed_variant=True,
            raise_error_on_existed_variant=False)

        return frame_variant_collection

    def test(self):
        frame_variants_collection = self.__create_frames_variants_collection()
        stemmer = MystemWrapper()
        p = LemmasBasedFrameVariantsParser(save_lemmas=False,
                                           stemmer=stemmer,
                                           frame_variants=frame_variants_collection)

        ctx = PipelineContext(d={"src": "мы пытались его осудить но не получилось".split()})

        p.apply(ctx)

        for t in ctx.provide("src"):
            s = "[{}]".format(t.Variant.get_value()) if isinstance(t, TextFrameVariant) else t
            print(s)


if __name__ == '__main__':
    unittest.main()
