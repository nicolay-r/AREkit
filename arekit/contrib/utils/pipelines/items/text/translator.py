from googletrans import Translator

from arekit.common.entities.base import Entity
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem


class TextAndEntitiesGoogleTranslator(BasePipelineItem):
    """ Text translator, based on GoogleTraslate service

        NOTE: Considered to be adopted right-after entities parsed
        but before the input tokenization into list of terms.

        NOTE#2: Move this pipeline item as a separated github project.
    """

    def __init__(self, src, dest):
        assert(isinstance(src, str))
        assert(isinstance(src, str))
        self.translator = Translator()
        self.__src = src
        self.__dest = dest

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert(isinstance(input_data, list))

        entities = []
        content = []
        parts_to_join = []

        for _, part in enumerate(input_data):
            if isinstance(part, str):
                parts_to_join.append(part)
            elif isinstance(part, Entity):
                content.append(" ".join(parts_to_join))
                content.append(part.Value)
                parts_to_join.clear()
                entities.append(part)

        if len(parts_to_join) > 0:
            content.append(" ".join(parts_to_join))

        # Compose text parts.
        translated_parts = [part.text for part in
                            self.translator.translate(content, dest=self.__dest, src=self.__src)]

        # NOTE: entities always are 1, 3, 5, 7 ... indexed
        for part_index in range(len(translated_parts)):
            if part_index % 2 == 0:
                continue
            # Pick up the related entity.
            entity = entities[int((part_index-1)/2)]
            entity.set_caption(translated_parts[part_index])

            translated_parts[part_index] = entity

        return translated_parts
