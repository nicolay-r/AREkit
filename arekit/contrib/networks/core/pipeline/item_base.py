from arekit.common.pipeline.item import BasePipelineItem
from arekit.contrib.networks.core.model_ctx import TensorflowModelContext


class EpochHandlingPipelineItem(BasePipelineItem):

    def __init__(self):
        self._context = None
        self._data_type = None

    @property
    def DataType(self):
        return self._data_type

    @property
    def ModelContext(self):
        return self._context

    def before_epoch(self, model_context, data_type):
        assert(isinstance(model_context, TensorflowModelContext))
        self._context = model_context
        self._data_type = data_type
