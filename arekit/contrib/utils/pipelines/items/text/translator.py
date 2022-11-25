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

        def __optionally_register(prts_to_join):
            if len(prts_to_join) > 0:
                content.append(" ".join(prts_to_join))
            parts_to_join.clear()

        content = []
        origin_entities = []
        origin_entity_ind = []
        parts_to_join = []

        for _, part in enumerate(input_data):
            if isinstance(part, str) and part.strip():
                parts_to_join.append(part)
            elif isinstance(part, Entity):
                # Register first the prior parts were merged.
                __optionally_register(parts_to_join)
                # Register entities information for further restoration.
                origin_entity_ind.append(len(content))
                origin_entities.append(part)
                content.append(part.Value)

        __optionally_register(parts_to_join)

        # Compose text parts.
        translated_parts = [part.text for part in
                            self.translator.translate(content, dest=self.__dest, src=self.__src)]

        for entity_ind, entity_part_ind in enumerate(origin_entity_ind):
            entity = origin_entities[entity_ind]
            entity.set_display_value(translated_parts[entity_part_ind])
            translated_parts[entity_part_ind] = entity

        return translated_parts
