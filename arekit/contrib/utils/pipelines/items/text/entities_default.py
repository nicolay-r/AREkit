from arekit.common.entities.base import Entity
from arekit.common.pipeline.items.base import BasePipelineItem


class TextEntitiesParser(BasePipelineItem):

    def __init__(self):
        super(TextEntitiesParser, self).__init__()

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
