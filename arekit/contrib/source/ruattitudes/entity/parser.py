from arekit.common.news.objects_parser import SentenceObjectsParserPipelineItem
from arekit.common.pipeline.context import PipelineContext
from arekit.common.text.partitioning.terms import TermsPartitioning
from arekit.contrib.source.ruattitudes.sentence import RuAttitudesSentence
from arekit.contrib.source.ruattitudes.text_object import TextObject


class RuAttitudesTextEntitiesParser(SentenceObjectsParserPipelineItem):

    KEY = "sentence"

    def __init__(self):
        super(RuAttitudesTextEntitiesParser, self).__init__(TermsPartitioning())

    # region protected methods

    def _get_text(self, pipeline_ctx):
        sentence = self.__get_sentence(pipeline_ctx)
        return sentence.Text

    def _get_parts_provider_func(self, input_data, pipeline_ctx):
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
        assert(isinstance(sentence, RuAttitudesSentence))
        for text_object in sentence.iter_objects():
            assert(isinstance(text_object, TextObject))
            entity = text_object.to_entity(lambda sent_id: sentence.get_doc_level_text_object_id(sent_id))
            yield entity, text_object.Bound

    # endregion
