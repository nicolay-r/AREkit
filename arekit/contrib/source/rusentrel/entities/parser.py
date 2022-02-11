from arekit.common.news.objects_parser import SentenceObjectsParserPipelineItem
from arekit.common.pipeline.context import PipelineContext
from arekit.common.text.partitioning.str import StringPartitioning
from arekit.contrib.source.rusentrel.sentence import RuSentRelSentence


class RuSentRelTextEntitiesParser(SentenceObjectsParserPipelineItem):

    KEY = "sentence"

    def __init__(self):
        super(RuSentRelTextEntitiesParser, self).__init__(StringPartitioning())

    # region protected methods

    def _get_text(self, pipeline_ctx):
        sentence = self.__get_sentence(pipeline_ctx)
        return sentence.Text

    def _get_parts_provider_func(self, pipeline_ctx):
        sentence = self.__get_sentence(pipeline_ctx)
        return self.__iter_subs_values_with_bounds(sentence)

    # endregion

    # region private methods

    def __get_sentence(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert(self.KEY in pipeline_ctx)
        return pipeline_ctx.provide(self.KEY)

    @staticmethod
    def __iter_subs_values_with_bounds(sentence):
        assert(isinstance(sentence, RuSentRelSentence))
        return sentence.iter_entity_with_local_bounds()

    # endregion
