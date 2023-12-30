import unittest

from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.batching import BatchingPipeline
from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.utils.pipelines.items.text.entities_default import TextEntitiesParser


class TestNestedEntities(unittest.TestCase):

    s = """24 марта президент [США] [Джо-Байден] провел переговоры с
           лидерами стран [Евросоюза] в [Брюсселе] , вызвав внимание рынка и предположения о
           том, что [Америке] удалось уговорить [ЕС] совместно бойкотировать российские нефть
           и газ.  [[Европейский]-[Союз]] крайне зависим от [России] в плане поставок нефти и
           газа."""

    def test(self):

        ctx = BasePipeline.run(pipeline=[TextEntitiesParser()],
                               pipeline_ctx=PipelineContext({"result": self.s.split()}))
        parsed_text = ctx.provide("result")

        print(parsed_text)

    def test_batched(self):

        # Compose a single batch with two sentences.
        ctx = BatchingPipeline.run(
            pipeline=[TextEntitiesParser()],
            pipeline_ctx=PipelineContext({"result": [self.s.split(), self.s.split()]}))
        parsed_text = ctx.provide("result")

        print(parsed_text)


if __name__ == '__main__':
    unittest.main()
