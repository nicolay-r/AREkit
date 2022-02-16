from arekit.common.entities.base import Entity
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.item import BasePipelineItem


class TextEntitiesParser(BasePipelineItem):

    def __init__(self):
        super(TextEntitiesParser, self).__init__()

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))

        # extract terms.
        words = pipeline_ctx.provide("src")
        assert(isinstance(words, list))

        # update the result.
        pipeline_ctx.update("src", value=[self.__process_word(w) for w in words])

    def __process_word(self, word):
        assert(isinstance(word, str))

        # If this is a special word which is related to the [entity] mention.
        if word[0] == "[" and word[-1] == "]":
            entity = Entity(value=word[1:-1], e_type=None)
            return entity

        return word
