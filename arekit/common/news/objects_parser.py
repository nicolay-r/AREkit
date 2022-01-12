from arekit.common.bound import Bound
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.item import BasePipelineItem


class SentenceObjectsParserPipelineItem(BasePipelineItem):

    def __init__(self, iter_objs_func):
        assert(callable(iter_objs_func))
        self.__iter_objs_func = iter_objs_func

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert("sentence" in pipeline_ctx)

        sentence = pipeline_ctx.provide("sentence")

        start = 0
        entries = []

        for value, bound in self.__iter_objs_func(sentence):
            assert(isinstance(bound, Bound))
            assert(bound.Position >= start)

            # Release everything till the current value position.
            part = sentence.Text[start:bound.Position]

            if isinstance(part, str):
                entries.append(part)
            else:
                entries.extend(part)

            # Release the entity value.
            entries.extend([value])

            start = bound.Position + bound.Length

        # Release everything after the last entity.
        last_part = sentence.Text[start:len(sentence.Text)]
        entries.extend([last_part])

        # update information in pipeline
        pipeline_ctx.update("src", entries)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass