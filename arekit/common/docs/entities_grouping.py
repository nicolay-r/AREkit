from arekit.common.entities.base import Entity
from arekit.common.pipeline.items.base import BasePipelineItem


class EntitiesGroupingPipelineItem(BasePipelineItem):

    def __init__(self, value_to_group_id_func):
        assert(callable(value_to_group_id_func))
        self.__value_to_group_id_func = value_to_group_id_func

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, list))

        for entity in filter(lambda term: isinstance(term, Entity), input_data):
            group_index = self.__value_to_group_id_func(entity.Value)
            entity.set_group_index(group_index)
