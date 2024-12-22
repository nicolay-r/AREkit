import unittest

from arekit.common.entities.base import Entity
from arekit.common.pipeline.base import BasePipelineLauncher
from arekit.common.pipeline.batching import BatchingPipelineLauncher
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem


class TextEntitiesParser(BasePipelineItem):

    def __init__(self, **kwargs):
        super(TextEntitiesParser, self).__init__(**kwargs)

    @staticmethod
    def __process_word(word):
        assert(isinstance(word, str))

        # If this is a special word which is related to the [entity] mention.
        if word[0] == "[" and word[-1] == "]":
            entity = Entity(value=word[1:-1], e_type=None)
            return entity

        return word

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, list))
        return [self.__process_word(w) for w in input_data]


class TestNestedEntities(unittest.TestCase):

    s = """24 марта президент [США] [Джо-Байден] провел переговоры с
           лидерами стран [Евросоюза] в [Брюсселе] , вызвав внимание рынка и предположения о
           том, что [Америке] удалось уговорить [ЕС] совместно бойкотировать российские нефть
           и газ.  [[Европейский]-[Союз]] крайне зависим от [России] в плане поставок нефти и
           газа."""

    def test(self):

        ctx = BasePipelineLauncher.run(pipeline=[TextEntitiesParser()],
                                       pipeline_ctx=PipelineContext({"result": self.s.split()}))
        parsed_text = ctx.provide("result")

        print(parsed_text)

    def test_batched(self):

        # Compose a single batch with two sentences.
        ctx = BatchingPipelineLauncher.run(
            pipeline=[TextEntitiesParser()],
            pipeline_ctx=PipelineContext({"result": [self.s.split(), self.s.split()]}))
        parsed_text = ctx.provide("result")

        print(parsed_text)


if __name__ == '__main__':
    unittest.main()
