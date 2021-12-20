from arekit.common.frames.text_variant import TextFrameVariant
from arekit.common.pipeline.context import PipelineContext
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.text.pipeline_frames_lemmatized import LemmasBasedFrameVariantsParser
from examples.repository import create_frame_variants_collection


if __name__ == '__main__':

    frame_variants_collection = create_frame_variants_collection()
    stemmer = MystemWrapper()
    p = LemmasBasedFrameVariantsParser(save_lemmas=False,
                                       stemmer=stemmer,
                                       frame_variants=frame_variants_collection)

    ctx = PipelineContext(d={"src": "мы пытались его осудить но не получилось".split()})

    p.apply(ctx)

    for t in ctx.provide("src"):
        s = "[{}]".format(t.Variant.get_value()) if isinstance(t, TextFrameVariant) else t
        print(s)
