from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.networks.core.feeding.batch.base import MiniBatch
from arekit.contrib.networks.core.pipeline.item_base import EpochHandlingPipelineItem


class EpochLabelsPredictorPipelineItem(EpochHandlingPipelineItem):

    KEY = "uint_labels"

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, MiniBatch))
        assert(isinstance(pipeline_ctx, PipelineContext))
        feed_dict = self._context.create_feed_dict(minibatch=input_data, data_type=self._data_type)
        uint_labels = self._context.Session.run(self._context.Network.Labels, feed_dict=feed_dict)
        pipeline_ctx.update(param=self.KEY, value=uint_labels)
